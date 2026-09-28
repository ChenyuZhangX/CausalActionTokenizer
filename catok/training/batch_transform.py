"""
Batch Transform：在 Dataset 加载时对 action 做 normalize 和 pad。

目的：
  1. 在 __getitem__ 阶段就完成 normalize，每个数据集用各自的 stats，避免混用
  2. 统一 pad 到 target_action_dim，解决不同数据集 action 维度不一致无法 batch 的问题

Bridge 专用：最后一维 0/1 -> [-1, 1]；其余维用 (2*(t-q01)/(q99-q01)-1).clip(-1,1)
"""

import json
from typing import Optional, Union

import numpy as np
from torch.utils.data import Dataset

from catok.training.statistics import build_action_normalizer


# =============================================================================
# Bridge 专用 Normalizer
# =============================================================================

class BridgeActionNormalizer:
    """
    Bridge 数据集的 action 归一化：
    - 最后一维（gripper 0/1）：映射到 [-1, 1]
    - 其余维度：2 * (t - q01) / (q99 - q01) - 1，再 clip(-1, 1)
    """

    def __init__(self, stats_path: str, eps: float = 1e-7):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        action_stats = stats["action"]
        self.q01 = np.array(action_stats["q01"], dtype=np.float32)
        self.q99 = np.array(action_stats["q99"], dtype=np.float32)
        self.eps = eps

    def normalize(self, action: np.ndarray) -> np.ndarray:
        out = action.copy()
        n_dim = out.shape[-1]
        if n_dim < 2:
            # 只有一维时，按最后一维处理
            out = np.clip(2 * out - 1, -1.0, 1.0).astype(np.float32)
            return out

        # 除最后一维外：2 * (t - q01) / (q99 - q01) - 1，clip(-1, 1)
        q01 = self.q01[: n_dim - 1]
        q99 = self.q99[: n_dim - 1]
        scale = q99 - q01 + self.eps
        out[..., :-1] = np.clip(
            2 * (out[..., :-1] - q01) / scale - 1,
            -1.0,
            1.0,
        )

        # 最后一维：0/1 -> [-1, 1]
        out[..., -1] = np.clip(2 * out[..., -1] - 1, -1.0, 1.0)

        return out.astype(np.float32)

    def denormalize(self, action: np.ndarray) -> np.ndarray:
        out = action.copy()
        n_dim = out.shape[-1]
        if n_dim < 2:
            out = (out + 1) / 2
            return out.astype(np.float32)

        q01 = self.q01[: n_dim - 1]
        q99 = self.q99[: n_dim - 1]
        out[..., :-1] = (out[..., :-1] + 1) / 2 * (q99 - q01) + q01
        out[..., -1] = (out[..., -1] + 1) / 2
        return out.astype(np.float32)


# =============================================================================
# 核心 Transform
# =============================================================================

def _pad_horizon(action: np.ndarray, target_horizon: int) -> np.ndarray:
    """Pad horizon dimension (second-to-last) to target_horizon."""
    current = action.shape[-2]
    if current >= target_horizon:
        return action[..., :target_horizon, :] if current > target_horizon else action
    pad_shape = list(action.shape)
    pad_shape[-2] = target_horizon - current
    return np.concatenate([action, np.zeros(pad_shape, dtype=action.dtype)], axis=-2)


def normalize_and_pad_action(
    action: np.ndarray,
    normalizer,
    target_action_dim: int,
    target_horizon: Optional[int] = None,
    action_start: int = 0,
) -> np.ndarray:
    """
    对 action 做 normalize，再 slice/pad 到 target_action_dim 和 target_horizon。

    Args:
        action: (..., horizon, action_dim) float32
        normalizer: ActionNormalizerBase 实例，若为 None 则跳过 normalize
        target_action_dim: 目标 action 维度
        target_horizon: 目标 horizon 长度，不足则尾部补 0；None 表示不 pad horizon
        action_start: 从第几个 dim 开始截取，默认 0

    Returns:
        (..., target_horizon, target_action_dim) float32
    """
    if normalizer is not None:
        out = normalizer.normalize(action.copy())  # copy 防止 in-place 修改原数据
    else:
        out = action

    # slice action_dim from action_start
    out = out[..., action_start:action_start + target_action_dim]

    # pad horizon（倒数第二维）
    if target_horizon is not None:
        current_horizon = out.shape[-2]
        if current_horizon < target_horizon:
            pad_shape = list(out.shape)
            pad_shape[-2] = target_horizon - current_horizon
            pad = np.zeros(pad_shape, dtype=out.dtype)
            out = np.concatenate([out, pad], axis=-2)
        elif current_horizon > target_horizon:
            out = out[..., :target_horizon, :]

    return out.astype(np.float32)


# =============================================================================
# Dataset 包装器
# =============================================================================

class ActionTransformDataset(Dataset):
    """
    包装一个返回 action 的 Dataset，应用 normalize + pad。

    每个数据集使用自己的 normalizer（对应自己的 stats_path），避免混 stats。

    性能优化：当底层 dataset 是 RLDSStateActionDataset（数据全在内存）时，
    在构造时一次性对所有 episode 做 normalize + pad（预计算），
    __getitem__ 直接返回 slice，避免逐样本重复计算。
    """

    def __init__(
        self,
        dataset: Dataset,
        normalizer=None,
        target_action_dim: int = 8,
        target_horizon: Optional[int] = None,
        action_only: bool = True,
        action_start: int = 0,
    ):
        """
        Args:
            dataset: 底层 Dataset，__getitem__ 返回 action 或 (state, action)
            normalizer: ActionNormalizerBase，None 表示不 normalize
            target_action_dim: 输出 action 的目标维度
            target_horizon: 输出 action 的 pad horizon 长度，None 表示不 pad
            action_only: 底层 dataset 是否只返回 action（否则返回 (state, action)）
            action_start: 从第几个 dim 开始截取，默认 0（兼容原行为）
        """
        self.dataset = dataset
        self.normalizer = normalizer
        self.target_action_dim = target_action_dim
        self.target_horizon = target_horizon
        self.action_only = action_only
        self.action_start = action_start
        self._precomputed = False

        # 预计算：对 RLDSStateActionDataset 的 episode_actions 做一次性 normalize + pad
        self._try_precompute()

    def _try_precompute(self):
        """尝试对内存中的 episode 数据做一次性 normalize + action_dim pad。

        normalize 和 action_dim pad 是逐时间步的，可以对整个 episode 做一次。
        horizon pad 依赖滑动窗口 slice，仍需在 __getitem__ 中处理。
        """
        from catok.training.rlds_dataset import RLDSStateActionDataset
        ds = self.dataset
        if not isinstance(ds, RLDSStateActionDataset):
            return

        # 对每个 episode 的 actions 做 normalize + action_dim pad（不做 horizon pad）
        for ep_idx in range(len(ds.episode_actions)):
            ep_actions = ds.episode_actions[ep_idx].astype(np.float32)
            ep_actions = normalize_and_pad_action(
                ep_actions, self.normalizer, self.target_action_dim,
                target_horizon=None,  # horizon pad 在 __getitem__ 中处理
                action_start=self.action_start,
            )
            ds.episode_actions[ep_idx] = ep_actions

        self._precomputed = True

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Union[np.ndarray, tuple]:
        if self._precomputed:
            # 预计算模式：normalize + action_dim pad 已完成，直接取 slice
            x = self.dataset[idx]
            if self.target_horizon is not None:
                action = x if self.action_only else x[1]
                action = _pad_horizon(action, self.target_horizon)
                if self.action_only:
                    return action
                return x[0], action
            return x

        x = self.dataset[idx]

        if self.action_only:
            return normalize_and_pad_action(
                x, self.normalizer, self.target_action_dim,
                self.target_horizon,
                action_start=self.action_start,
            )

        state, action = x
        action_out = normalize_and_pad_action(
            action, self.normalizer, self.target_action_dim,
            self.target_horizon,
            action_start=self.action_start,
        )
        return state, action_out


# =============================================================================
# 工厂函数
# =============================================================================

def build_action_transform(
    stats_path: Optional[str] = None,
    normalizer_method: str = "qq",
    normalizer_config: Optional[dict] = None,
) -> Optional[object]:
    """
    根据 stats_path 构建 normalizer，用于 ActionTransformDataset。

    Args:
        stats_path: stats JSON 路径，None 则返回 None（不 normalize）
        normalizer_method: "qq", "zscore", "bridge" 等
            - "bridge": 最后一维 0/1->[-1,1]，其余维 2*(t-q01)/(q99-q01)-1
        normalizer_config: 额外配置，如 clip, action_dim 等

    Returns:
        normalizer 实例或 None
    """
    if stats_path is None:
        return None

    if normalizer_method == "bridge":
        return BridgeActionNormalizer(stats_path)

    config = dict(normalizer_config) if normalizer_config else {}
    config["stats_path"] = stats_path
    return build_action_normalizer(normalizer_method, config)


def wrap_dataset_with_transform(
    dataset: Dataset,
    stats_path: Optional[str] = None,
    target_action_dim: int = 8,
    target_horizon: Optional[int] = None,
    action_only: bool = True,
    normalizer_method: str = "qq",
    normalizer_config: Optional[dict] = None,
    action_start: int = 0,
) -> ActionTransformDataset:
    """
    用 normalize + slice + pad 包装 dataset。

    Args:
        dataset: 底层 Dataset
        stats_path: 该数据集对应的 stats 路径，None 则不 normalize
        target_action_dim: 目标 action 维度（slice 后的维度）
        target_horizon: pad 目标 horizon 长度，None 表示不 pad
        action_only: dataset 是否只返回 action
        normalizer_method: normalizer 类型
        normalizer_config: normalizer 额外配置
        action_start: 从第几个 dim 开始截取，默认 0

    Returns:
        ActionTransformDataset
    """
    normalizer = build_action_transform(
        stats_path=stats_path,
        normalizer_method=normalizer_method,
        normalizer_config=normalizer_config,
    )
    return ActionTransformDataset(
        dataset=dataset,
        normalizer=normalizer,
        target_action_dim=target_action_dim,
        target_horizon=target_horizon,
        action_only=action_only,
        action_start=action_start,
    )
