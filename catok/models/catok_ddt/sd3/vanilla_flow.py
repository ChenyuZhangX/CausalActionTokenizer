import torch
import torch.nn.functional as F

from catok.models.catok_ddt.vanilla_utils import calc_loss, get_mask

class VanillaFlow:
    def __init__(
        self,
        model,
        context_see_xt=True,
        is_causal=True,
        noise_level=1.0,
        use_diti=False,
        diti=None,
        t2k=1.0,
        diti_input_mode="auto",
        consistency_weight=0.0,
        consistency_delta_t=0.05,
        consistency_detach_target=True,
        use_logit_normal=False,
        logit_normal_mean=0.0,
        logit_normal_std=1.0,
        logit_normal_mix_ratio=0.0,
    ):
        self.model = model
        self.context_see_xt = context_see_xt
        self.is_causal = is_causal
        self.noise_level = noise_level
        self.use_diti = use_diti
        self.diti = diti
        self.t2k = t2k
        self.diti_input_mode = diti_input_mode
        self.consistency_weight = consistency_weight
        self.consistency_delta_t = consistency_delta_t
        self.consistency_detach_target = consistency_detach_target
        self.use_logit_normal = use_logit_normal
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.logit_normal_mix_ratio = logit_normal_mix_ratio

    def _to_diti_time(self, t):
        t_scaled = (self.t2k * t).clamp(0.0, 1.0)
        if self.diti_input_mode == "normalized":
            return t_scaled
        if self.diti_input_mode == "scaled_1000":
            return t_scaled * 1000.0
        if self.diti_input_mode != "auto":
            raise ValueError(f"Unsupported diti_input_mode: {self.diti_input_mode}")

        # Auto mode:
        # - DiTi/DiTi_cont expect 0..1000
        # - DiTi_normal expects 0..1
        diti_name = self.diti.__class__.__name__
        if diti_name == "DiTi_normal":
            return t_scaled
        return t_scaled * 1000.0

    def _get_cond_mask(self, t, n_tokens, full_tokens=False):
        if full_tokens:
            return torch.ones(t.shape[0], n_tokens, dtype=torch.bool, device=t.device)

        if not self.use_diti or self.diti is None:
            return get_mask(t, n_tokens=n_tokens, is_causal=self.is_causal)

        k = self.diti.to_indices(self._to_diti_time(t)).clamp(0, n_tokens - 1)
        # Keep the original get_mask behavior by converting k back to an
        # equivalent normalized threshold.
        t_proxy = ((k.to(t.dtype) + 1.0) / float(n_tokens)).clamp(0.0, 1.0)
        return get_mask(t_proxy, n_tokens=n_tokens, is_causal=self.is_causal)

    def get_loss(self, x_1, encoder_hidden_states, padding_mask=None):
        """
        x_1: 真实数据 (Data)
        x_0: 噪声 (Noise)
        t: 时间 [0, 1]
        padding_mask: bool tensor (B, H, A), True=valid, False=padded. Optional.
        """
        b = x_1.shape[0]

        # 1. 采样 x_0 (纯高斯噪声)
        x_0 = torch.randn_like(x_1) * self.noise_level

        # 2. 采样时间 t
        if self.use_logit_normal:
            # Logit-normal: u ~ N(mean, std), t = sigmoid(u)
            # 混合采样：mix_ratio 比例的样本使用 uniform，保证 t≈0/1 区域有足够训练密度
            u = torch.randn(b, device=x_1.device) * self.logit_normal_std + self.logit_normal_mean
            t = torch.sigmoid(u)
            if self.logit_normal_mix_ratio > 0.0:
                use_uniform = torch.rand(b, device=x_1.device) < self.logit_normal_mix_ratio
                t_uniform = torch.rand(b, device=x_1.device)
                t = torch.where(use_uniform, t_uniform, t)
        else:
            t = torch.rand(b, device=x_1.device)

        # 3. 线性插值 (Linear Interpolation)
        # x_t = t * x_1 + (1 - t) * x_0
        # reshape t for broadcasting
        t_img = t.view(b, 1, 1)
        x_t = t_img * x_1 + (1 - t_img) * x_0

        # 4. 计算目标速度 (Target Velocity)
        # flow 直线导数: d/dt (t*x1 + (1-t)*x0) = x_1 - x_0
        target_v = x_1 - x_0

        # 5. 模型预测
        pred_v, _ = self.model(
            x_t,
            t,
            encoder_hidden_states=encoder_hidden_states,
            context_see_xt=self.context_see_xt,
            mask=self._get_cond_mask(t, n_tokens=encoder_hidden_states.shape[1]),
        )
        recon_x = x_t + (1 - t_img) * pred_v
        
        # 6. MSE Loss
        loss, loss_dict = calc_loss(pred_v, target_v, dct_loss=True, loss_fn=F.l1_loss, padding_mask=padding_mask)

        # 7. Self-consistency loss (optional)
        # Encourage velocity predictions to stay consistent after a small forward step.
        if self.consistency_weight > 0:
            # Random small step in time; keep shape for broadcasting.
            delta_t = torch.rand_like(t) * self.consistency_delta_t
            t_next = (t + delta_t).clamp(max=1.0)
            dt_img = (t_next - t).view(b, 1, 1)

            if self.consistency_detach_target:
                x_t_next = x_t + dt_img * pred_v.detach()
                v_target = pred_v.detach()
            else:
                x_t_next = x_t + dt_img * pred_v
                v_target = pred_v

            pred_v_next, _ = self.model(
                x_t_next,
                t_next,
                encoder_hidden_states=encoder_hidden_states,
                context_see_xt=self.context_see_xt,
                mask=self._get_cond_mask(t_next, n_tokens=encoder_hidden_states.shape[1]),
            )
            consistency_loss = F.mse_loss(pred_v_next, v_target)
            loss = loss + self.consistency_weight * consistency_loss
            loss_dict["consistency"] = consistency_loss.item()
        else:
            loss_dict["consistency"] = 0.0

        return loss, loss_dict, recon_x

    @torch.no_grad()
    def _predict_v(self, x, t_batch, encoder_hidden_states, full_tokens=False):
        v_pred, _ = self.model(
            x,
            t_batch,
            encoder_hidden_states=encoder_hidden_states,
            context_see_xt=self.context_see_xt,
            mask=self._get_cond_mask(
                t_batch,
                n_tokens=encoder_hidden_states.shape[1],
                full_tokens=full_tokens,
            ),
        )
        return v_pred

    @torch.no_grad()
    def euler_steps(self, x, encoder_hidden_states, time_grid, full_tokens=False):
        b = x.shape[0]
        device = x.device
        total_steps = time_grid.shape[0] - 1

        for i in range(total_steps):
            t_now = time_grid[i]
            t_next = time_grid[i + 1]
            dt = t_next - t_now
            t_batch = torch.full((b,), t_now, device=device)

            v_pred = self._predict_v(x, t_batch, encoder_hidden_states, full_tokens=full_tokens)
            x = x + v_pred * dt

        return x

    @torch.no_grad()
    def heun_steps(self, x, encoder_hidden_states, time_grid, full_tokens=False):
        b = x.shape[0]
        device = x.device
        total_steps = time_grid.shape[0] - 1

        for i in range(total_steps):
            t_now = time_grid[i]
            t_next = time_grid[i + 1]
            dt = t_next - t_now

            t_now_batch = torch.full((b,), t_now, device=device)
            t_next_batch = torch.full((b,), t_next, device=device)

            # Predictor
            k1 = self._predict_v(x, t_now_batch, encoder_hidden_states, full_tokens=full_tokens)
            x_pred = x + dt * k1
            # Corrector
            k2 = self._predict_v(x_pred, t_next_batch, encoder_hidden_states, full_tokens=full_tokens)
            x = x + 0.5 * dt * (k1 + k2)

        return x

    @torch.no_grad()
    def sample(
        self,
        shape,
        encoder_hidden_states,
        steps=20,
        one_step=False,
        solver="euler",
        full_tokens=False,
    ):
        """
        使用数值积分求解 ODE（支持 Euler / Heun）
        从 t=0 (Noise) 积分到 t=1 (Data)
        """
        self.model.eval()
        device = encoder_hidden_states.device
        
        # 从噪声开始
        # print("self.noise_level:", self.noise_level)
        # breakpoint()
        x = torch.randn(shape, device=device) * self.noise_level

        total_steps = 1 if one_step else steps
        if total_steps <= 0:
            raise ValueError(f"steps must be positive, got {total_steps}")

        # 生成时间步序列: 0 -> 1 (包含终点，便于计算每步 dt)
        time_grid = torch.linspace(0, 1, total_steps + 1, device=device)

        if solver == "euler":
            x = self.euler_steps(
                x,
                encoder_hidden_states,
                time_grid,
                full_tokens=full_tokens,
            )
        elif solver == "heun":
            x = self.heun_steps(
                x,
                encoder_hidden_states,
                time_grid,
                full_tokens=full_tokens,
            )
        else:
            raise ValueError(f"Unsupported solver: {solver}. Choose from ['euler', 'heun'].")

        self.model.train()
        return x


class RectifiedFlow:
    def __init__(
        self,
        model,
        context_see_xt=True,
        is_causal=True,
        noise_level=1.0,
        num_timesteps=100,
        start=1.0,
        val_schedule="shift",
        shift=1.0,
    ):
        self.model = model
        self.context_see_xt = context_see_xt
        self.is_causal = is_causal
        self.noise_level = noise_level

        self.shift = shift
        self.num_timesteps = num_timesteps
        self.start = start
        self.val_schedule = val_schedule

    def _schedule(self, t):
        if self.val_schedule == "uniform":
            return t
        if self.val_schedule == "shift":
            return self.shift * t / (1 + (self.shift - 1) * t)
        raise NotImplementedError(f"Unsupported val_schedule: {self.val_schedule}")

    def get_loss(self, x_1, encoder_hidden_states, padding_mask=None):
        """
        padding_mask: bool tensor (B, H, A), True=valid, False=padded. Optional.
        """
        b = x_1.shape[0]
        x_noise = torch.randn_like(x_1) * self.noise_level

        # RF training path: x_t = t * noise + (1 - t) * data
        t = torch.rand(b, device=x_1.device)
        t_img = t.view(b, 1, 1)
        x_t = t_img * x_noise + (1 - t_img) * x_1
        target_v = x_noise - x_1

        pred_v, _ = self.model(
            x_t,
            t,
            encoder_hidden_states=encoder_hidden_states,
            context_see_xt=self.context_see_xt,
            mask=get_mask(
                t,
                n_tokens=encoder_hidden_states.shape[1],
                is_causal=self.is_causal,
            ),
        )
        recon_x = x_t - t_img * pred_v
        loss, loss_dict = calc_loss(pred_v, target_v, dct_loss=True, loss_fn=F.l1_loss, padding_mask=padding_mask)
        return loss, loss_dict, recon_x

    @torch.no_grad()
    def sample(self, shape, encoder_hidden_states, steps=None):
        self.model.eval()
        b = shape[0]
        device = encoder_hidden_states.device
        total_steps = self.num_timesteps if steps is None else steps

        # RF sampling starts from noise and integrates from t=1 -> t=0.
        x = torch.randn(shape, device=device) * self.noise_level
        base_t = torch.linspace(self.start, 0.0, total_steps + 1, device=device)
        scheduled_t = self._schedule(base_t)

        for i in range(total_steps):
            t_now = scheduled_t[i]
            t_next = scheduled_t[i + 1]
            dt = t_now - t_next

            t_batch = torch.full((b,), t_now, device=device)
            v_pred, _ = self.model(
                x,
                t_batch,
                encoder_hidden_states=encoder_hidden_states,
                context_see_xt=self.context_see_xt,
                mask=get_mask(
                    t_batch,
                    n_tokens=encoder_hidden_states.shape[1],
                    is_causal=self.is_causal,
                ),
            )
            x = x - dt * v_pred

        self.model.train()
        return x