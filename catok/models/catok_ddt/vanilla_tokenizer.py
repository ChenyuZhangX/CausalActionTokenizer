import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Optional

from catok.models.catok_ddt.modules import DualBlock
from catok.models.catok_ddt.sd3.mmdit import MMDiT_Action
from catok.models.catok_ddt.sd3.vanilla_flow import RectifiedFlow, VanillaFlow
from catok.models.catok_ddt.diti_utils import DiTi, DiTi_cont, DiTi_normal
from catok.models.catok_ddt.vanilla_utils import calc_loss
from catok.models.catok_ddt.vanilla_encoder import VanillaEncoder
from catok.models.catok_ddt.vanilla_vq import VectorQuantizer, VectorQuantizerEMA, ResidualVQEMA


def symmetric_infonce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE (ActionCodec Appendix A.2.1): align F(A) with F(A+η); batch negatives."""
    z1 = F.normalize(z1, dim=-1, eps=1e-8)
    z2 = F.normalize(z2, dim=-1, eps=1e-8)
    logits_12 = (z1 @ z2.transpose(0, 1)) / temperature
    logits_21 = logits_12.transpose(0, 1)
    targets = torch.arange(z1.shape[0], device=z1.device, dtype=torch.long)
    return 0.5 * (
        F.cross_entropy(logits_12, targets)
        + F.cross_entropy(logits_21, targets)
    )


class VanillaVQVAE(nn.Module):
    def __init__(
        self,
        input_dim=8,
        seq_len=10,
        latent_len=5,
        d_model=32,
        d_vq=16,
        n_e=128,
        num_layers=2,
        vq_mode: Optional[Literal['VQ', 'VQ_EMA', 'RVQ_EMA']] = None,
        vq_threshold: Optional[float] = None,
        vq_check_every: Optional[int] = None,
        encoder_mode: Optional[Literal['Transformer', 'Dual']] = 'Transformer'
    ):
        super().__init__()
        self.latent_len = latent_len
        
        self.enc_input = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Encoder
        self.encoder_mode = encoder_mode
        if encoder_mode == 'Transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=4*d_model, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.compress = nn.Linear(seq_len, latent_len)  # not use
        elif encoder_mode == 'Dual':
            self.num_tokens = latent_len
            self.query_tokens = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02) # mean=0.0, std=0.02
            self.encoder = nn.ModuleList(
                [DualBlock(
                    hidden_size=d_model,
                    num_heads=4,
                    mlp_ratio=4,
                    query_dim=d_model,
                    bidirectional=False,
                    time_adaln=True,
                    qk_norm=False,
                    diti=None,
                ) for _ in range(num_layers)]
            )
                
        # VQ
        self.vq_mode = vq_mode
        if vq_mode == 'VQ':
            self.vq = VectorQuantizer(num_embeddings=n_e, embedding_dim=d_model)
        elif vq_mode == 'VQ_EMA':
            self.vq = VectorQuantizerEMA(num_embeddings=n_e, embedding_dim=d_model, vq_dim=d_vq)
        elif vq_mode == 'RVQ_EMA':
            self.vq = ResidualVQEMA(num_quantizers=3, num_embeddings=n_e, embedding_dim=d_model, vq_dim=d_vq, threshold=vq_threshold, check_every=vq_check_every)
        elif vq_mode is None:
            pass

        # Decoder
        self.decompress = nn.Linear(latent_len, seq_len)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=4*d_model, batch_first=True)
        self.decoder_tf = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, input_dim)
    
    def forward(self, x):
        # x: (B, 10, 8)
        x = self.enc_input(x) + self.pos_embedding

        if self.encoder_mode == 'Transformer':
            z = self.encoder(x) # (B, 10, 32)
        elif self.encoder_mode == 'Dual':
            query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
            for block in self.encoder:
                x, query_tokens = block(x, query_tokens)
            z = query_tokens
        
        # quantize
        if self.vq_mode is not None:
            vq_loss, z_q, indices, perplexity = self.vq(z)
        else:
            vq_loss = torch.tensor(0.0, device=x.device)
            z_q = z
            indices = None
            perplexity = torch.tensor(0.0, device=x.device)
        
        # decode
        z_up = z_q
        if self.encoder_mode == 'Dual':
            z_up = z_q.transpose(1, 2)
            z_up = self.decompress(z_up).transpose(1, 2) # (B, 10, 32)

        out = self.decoder_tf(z_up)
        recon_x = self.output_layer(out)
        
        return recon_x, vq_loss, indices, perplexity

class VanillaVQDiffusion(nn.Module):
    def __init__(
        self,
        horizon=20,
        action_dim=8,
        latent_len=8,
        add_noise: bool = False,
        # encoder
        d_model=128,
        num_layers=4,
        encoder_mode: Optional[Literal['Transformer', 'Dual']] = 'Transformer',
        embedder_type='linear',
        add_vq_latent: bool = True,
        # vq
        n_e=128,
        d_vq=16,
        vq_mode: Optional[Literal['VQ', 'VQ_EMA', 'RVQ_EMA']] = None,
        vq_threshold: Optional[float] = None,
        vq_check_every: Optional[int] = None,
        # decoder
        context_see_xt: bool = True,
        noise_level: float = 1.0,
        is_causal: bool = True,
        flow_type: Optional[Literal['vanilla', 'rectified']] = 'vanilla',
        flow_cfg: dict = None,
        # ActionCodec (arXiv:2602.15397) Appendix A.2.1 — latent InfoNCE to raise overlap rate
        overlap_infonce_weight: float = 0.0,
        overlap_infonce_std: float = 0.02,
        overlap_infonce_temperature: float = 0.07,
        # Adjacent-window InfoNCE: pull encoder(x_t) and encoder(x_{t+1}) together
        adjacent_infonce_weight: float = 0.0,
        adjacent_infonce_temperature: float = 0.07,
        # OR direct loss: KL divergence between soft VQ assignments of adjacent windows
        or_direct_weight: float = 0.0,
        or_direct_temperature: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.is_causal = is_causal
        self.flow_type = flow_type
        self.vanilla_use_diti = kwargs.get('vanilla_use_diti', False)

        self.codebook_size = n_e
        self.num_tokens = latent_len

        self.overlap_infonce_weight = float(overlap_infonce_weight)
        self.overlap_infonce_std = float(overlap_infonce_std)
        self.overlap_infonce_temperature = float(overlap_infonce_temperature)
        self.adjacent_infonce_weight = float(adjacent_infonce_weight)
        self.adjacent_infonce_temperature = float(adjacent_infonce_temperature)
        self.or_direct_weight = float(or_direct_weight)
        self.or_direct_temperature = float(or_direct_temperature)

        # Encoder
        self.encoder = VanillaEncoder(
            horizon=horizon,
            action_dim=action_dim,
            d_model=d_model,
            num_layers=num_layers,
            encoder_mode=encoder_mode,
            embedder_type=embedder_type,
            latent_len=latent_len,
            # add_vq_latent=add_vq_latent
            **kwargs
        )
                
        # VQ
        self.vq_mode = vq_mode
        if vq_mode == 'VQ':
            self.vq = VectorQuantizer(num_embeddings=n_e, embedding_dim=d_model, vq_dim=d_vq)
        elif vq_mode == 'VQ_EMA':
            self.vq = VectorQuantizerEMA(num_embeddings=n_e, embedding_dim=d_model, threshold=vq_threshold, check_every=vq_check_every, add_vq_latent=add_vq_latent, vq_dim=d_vq)
        elif vq_mode == 'RVQ_EMA':
            self.vq = ResidualVQEMA(num_quantizers=3, num_embeddings=n_e, embedding_dim=d_model, threshold=vq_threshold, check_every=vq_check_every, add_vq_latent=add_vq_latent)
        elif vq_mode is None:
            pass

        # Decoder
        self.context_see_xt = context_see_xt
        if embedder_type == 'patch':
            decoder_embedder_config = {
                'embedder_type': 'patch2',
                'channels_h': kwargs.get('channels_h', 1),
                'channels_a': kwargs.get('channels_a', 8),
            }
        else:
            decoder_embedder_config = {}
        self.decoder = MMDiT_Action(
            action_dim=action_dim,
            action_T=horizon,
            decoder_hidden_dim=d_model,
            depth=num_layers,
            K=latent_len,
            time_adaln='pos_emb',
            class_dropout_prob=0,
            train_filter=None,
            learnable_pos_embed=True,
            **decoder_embedder_config
        )

        self.vanilla_diti = None
        if self.flow_type == 'vanilla' and self.vanilla_use_diti:
            vanilla_diti_type = kwargs.get('vanilla_diti_type', 'uniform')
            if vanilla_diti_type == 'uniform':
                self.vanilla_diti = DiTi(
                    1000,
                    latent_len,
                    kwargs.get('vanilla_diti_stages', ''),
                    kwargs.get('vanilla_diti_k_per_stage', ''),
                )
            elif vanilla_diti_type == 'cont':
                self.vanilla_diti = DiTi_cont(
                    1000,
                    latent_len,
                    kwargs.get('vanilla_diti_stages', ''),
                    kwargs.get('vanilla_diti_k_per_stage', ''),
                )
            elif vanilla_diti_type == 'normal':
                self.vanilla_diti = DiTi_normal(
                    1000,
                    latent_len,
                    kwargs.get('vanilla_diti_m', 0.0),
                    kwargs.get('vanilla_diti_s', 1.0),
                )
            else:
                raise ValueError(
                    f"Unsupported vanilla_diti_type: {vanilla_diti_type}. "
                    "Choose from ['uniform', 'cont', 'normal']."
                )

        _flow_cfg = flow_cfg or {}
        if self.flow_type == 'vanilla':
            self.flow = VanillaFlow(
                self.decoder,
                context_see_xt=context_see_xt,
                is_causal=is_causal,
                noise_level=noise_level,
                use_diti=self.vanilla_use_diti,
                diti=self.vanilla_diti,
                t2k=kwargs.get('vanilla_t2k', 1.0),
                diti_input_mode=kwargs.get('vanilla_diti_input_mode', 'auto'),
                consistency_weight=kwargs.get('vanilla_consistency_weight', 0.0),
                consistency_delta_t=kwargs.get('vanilla_consistency_delta_t', 0.05),
                consistency_detach_target=kwargs.get('vanilla_consistency_detach_target', True),
                use_logit_normal=_flow_cfg.get('use_logit_normal', False),
                logit_normal_mean=_flow_cfg.get('logit_normal_mean', 0.0),
                logit_normal_std=_flow_cfg.get('logit_normal_std', 1.0),
                logit_normal_mix_ratio=_flow_cfg.get('logit_normal_mix_ratio', 0.0),
            )
        elif self.flow_type == 'rectified':
            self.flow = RectifiedFlow(
                self.decoder,
                context_see_xt=context_see_xt,
                is_causal=is_causal,
                noise_level=noise_level,
                num_timesteps=kwargs.get('rf_num_timesteps', 100),
                start=kwargs.get('rf_start', 1.0),
                val_schedule=kwargs.get('rf_val_schedule', 'shift'),
                shift=kwargs.get('rf_shift', 1.0),
            )
        else:
            raise ValueError(f"Unsupported flow_type: {self.flow_type}")
        self.add_noise = add_noise
        print("add_noise: ", self.add_noise)
        print("flow_type: ", self.flow_type)
    
    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """将输入 pad 到模型固定的 (self.horizon, self.action_dim)。

        Args:
            x: (B, H, A)，H <= self.horizon，A <= self.action_dim
        Returns:
            (B, self.horizon, self.action_dim)
        """
        B, H, A = x.shape
        if H < self.horizon:
            pad = torch.zeros(B, self.horizon - H, A, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        if A < self.action_dim:
            pad = torch.zeros(B, x.shape[1], self.action_dim - A, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        return x

    def forward(self, x, one_step=False, padding_mask=None, x_next=None):
        """
        Args:
            x: (B, horizon, action_dim) — 已由 dataloader pad 至目标维度
            padding_mask: bool tensor (B, horizon, action_dim), True=有效位置, False=padding。
                          传入时 loss 只计算有效位置，不传时行为与之前相同。
            x_next: (B, horizon, action_dim) optional — adjacent window for adjacent InfoNCE.
        """
        if self.add_noise:
            # uniform random from 0 to 0.2
            dist = torch.distributions.Uniform(0.0, 0.2)
            noise_level = dist.sample()
            noise = torch.randn_like(x) * noise_level
            noise[..., -2:] = torch.zeros_like(noise[..., -2:]).to(device=x.device, dtype=x.dtype)
            inputs = x.clone() + noise.detach()
        else:
            inputs = x
        # encode
        z = self.encoder(inputs)
        
        # quantize
        if self.vq_mode is not None:
            vq_loss, z_q, indices, perplexity = self.vq(z)
        else:
            vq_loss = torch.tensor(0.0, device=x.device)
            z_q = z
            indices = None
            perplexity = torch.tensor(0.0, device=x.device)
        
        # decode
        if one_step:
            recon_x, _ = self.decoder.forward_one_step_decoder(encoder_hidden_states=z_q, context_see_xt=self.context_see_xt)
            loss, loss_dict = calc_loss(recon_x, x, dct_loss=True, loss_fn=F.l1_loss, padding_mask=padding_mask)
            loss_dict['vq_loss'] = vq_loss.item()
        else:
            loss, loss_dict, recon_x = self.flow.get_loss(x, z_q, padding_mask=padding_mask)
            loss += vq_loss
            loss_dict['vq_loss'] = vq_loss.item()

        # Latent contrastive loss: pull F(A) and F(A+η) together (higher temporal overlap on codes).
        if (
            self.training
            and self.overlap_infonce_weight > 0.0
            and self.overlap_infonce_std > 0.0
        ):
            x_pad = self._pad_input(x)
            noise = torch.randn_like(x_pad) * self.overlap_infonce_std
            z_a = self.encoder(x_pad)
            z_p = self.encoder(x_pad + noise)
            emb_a = z_a.mean(dim=1)
            emb_p = z_p.mean(dim=1)
            infonce = symmetric_infonce(emb_a, emb_p, self.overlap_infonce_temperature)
            w = self.overlap_infonce_weight
            loss = loss + w * infonce
            loss_dict['overlap_infonce'] = infonce.item()

        # Compute adjacent encoder outputs once, shared by both InfoNCE and or_direct losses.
        need_adj = self.training and x_next is not None and (
            self.adjacent_infonce_weight > 0.0 or
            (self.or_direct_weight > 0.0 and self.vq_mode in ('VQ_EMA', 'VQ'))
        )
        if need_adj:
            xa_pad = self._pad_input(x)
            xp_pad = self._pad_input(x_next)
            za_enc = self.encoder(xa_pad)   # (B, K, d_model)
            zp_enc = self.encoder(xp_pad)   # (B, K, d_model)

        # Adjacent-window InfoNCE: pull F(x_t) and F(x_{t+1}) together directly.
        if need_adj and self.adjacent_infonce_weight > 0.0:
            adj_loss = symmetric_infonce(za_enc.mean(dim=1), zp_enc.mean(dim=1), self.adjacent_infonce_temperature)
            loss = loss + self.adjacent_infonce_weight * adj_loss
            loss_dict['adjacent_infonce'] = adj_loss.item()

        # OR direct loss: minimize KL divergence between soft VQ assignments of adjacent windows.
        # Soft assignment = softmax(-distances_to_codebook / tau), encourages same code selection.
        if need_adj and self.or_direct_weight > 0.0 and self.vq_mode in ('VQ_EMA', 'VQ'):
            za = self.vq.pre_proj(za_enc)   # (B, K, vq_dim)
            zp = self.vq.pre_proj(zp_enc)   # (B, K, vq_dim)
            emb = self.vq.embedding          # (n_e, vq_dim)
            def soft_assign(z, tau):
                B, K, D = z.shape
                zf = z.reshape(B * K, D)
                dist2 = (torch.sum(zf ** 2, dim=1, keepdim=True)
                         + torch.sum(emb ** 2, dim=1)
                         - 2 * zf @ emb.t())              # (B*K, n_e)
                return F.softmax(-dist2 / tau, dim=-1).reshape(B, K, -1)  # (B, K, n_e)
            p_a = soft_assign(za, self.or_direct_temperature)
            p_p = soft_assign(zp, self.or_direct_temperature)
            kl = (F.kl_div(p_a.log(), p_p, reduction='batchmean') +
                  F.kl_div(p_p.log(), p_a, reduction='batchmean')) * 0.5
            loss = loss + self.or_direct_weight * kl
            loss_dict['or_direct'] = kl.item()

        return recon_x, loss, loss_dict, indices, perplexity

    @torch.no_grad()
    def reconstruct(self, x, steps=20, one_step=False):
        orig_H, orig_A = x.shape[-2], x.shape[-1]
        x = self._pad_input(x)

        z = self.encoder(x)

        if self.vq_mode is not None:
            vq_loss, z_q, indices, perplexity = self.vq(z)
        else:
            z_q = z

        if one_step:
            recon_x, _ = self.decoder.forward_one_step_decoder(encoder_hidden_states=z_q, context_see_xt=self.context_see_xt)
        else:
            recon_x = self.flow.sample(x.shape, z_q, steps)

        return recon_x[:, :orig_H, :orig_A]
    
    @torch.no_grad()
    def encoding(self, x):
        """编码 action 序列为离散 token indices。

        自动将输入 pad 到模型的 (self.horizon, self.action_dim)。
        """
        x = self._pad_input(x)
        z = self.encoder(x)
        assert self.vq_mode is not None

        _, _, indices, _ = self.vq(z)
        # RVQ returns list of tensors, stack to (B, N_Codebook, Seq_len)
        if isinstance(indices, list):
            indices = torch.stack(indices, dim=1)
        return indices

    @torch.no_grad()
    def decoding(self, indices, steps=20, one_step=False, out_horizon=None, out_action_dim=None):
        """从 token indices 解码重建 action 序列。

        Args:
            out_horizon: 若指定，裁剪输出到此 horizon 长度（用于还原 pad 前的原始 horizon）。
            out_action_dim: 若指定，裁剪输出到此 action 维度（用于还原 pad 前的原始 action_dim）。
        """
        z_q = self.vq.decode(indices)

        if one_step:
            recon_x, _ = self.decoder.forward_one_step_decoder(encoder_hidden_states=z_q, context_see_xt=self.context_see_xt)
        else:
            recon_x = self.flow.sample((indices.shape[0], self.horizon, self.action_dim), z_q, steps)

        if out_horizon is not None:
            recon_x = recon_x[:, :out_horizon, :]
        if out_action_dim is not None:
            recon_x = recon_x[..., :out_action_dim]
        return recon_x

def count_parameters(model):
    # numel() 返回张量中元素的总数
    total_params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Buffers: {buffers:,}")
    print(f"Learnable Parameters: {trainable_params:,}")


if __name__ == '__main__':
    # model = VanillaVQVAE(vq_mode='VQ', d_model=128, num_layers=8, n_e=4096)
    # count_parameters(model)
    # model = VanillaVQVAE(vq_mode='VQ_EMA')
    # count_parameters(model)
    # model = VanillaVQVAE(vq_mode='RVQ_EMA', d_model=128, num_layers=8, n_e=4096, encoder_mode='Dual')
    model = VanillaVQDiffusion(vq_mode='RVQ_EMA', d_model=128, num_layers=8, n_e=4096, encoder_mode='Dual')
    count_parameters(model)