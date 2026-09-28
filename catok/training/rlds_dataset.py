"""
RLDS State-Action Dataset

基于 catok.utils.data_utils 的 load_rlds_state_action，
提供 PyTorch Dataset 接口，用于加载 RLDS 格式的 state 和 action（无图像）。

每个样本为 (state_window, action_window)，shape 均为 (horizon, dim)。
支持 action_only / state_only 模式。
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler, WeightedRandomSampler

from catok.utils.data_utils import load_rlds_state_action
from catok.training.batch_transform import wrap_dataset_with_transform


# =============================================================================
# DDP 兼容的加权 Sampler
# =============================================================================

class DistributedWeightedSampler(Sampler[int]):
    """
    加权采样 + DDP 分区：每个 rank 只采样自己分区的索引，按权重采样。
    需在每 epoch 开始时调用 set_epoch(epoch)。
    """

    def __init__(
        self,
        weights,
        num_samples: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.weights = weights
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

        if drop_last and num_samples % world_size != 0:
            self.num_samples = num_samples - num_samples % world_size
        self.num_samples_per_rank = self.num_samples // world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 打乱后分区（与 DistributedSampler 一致）
        shuffled = torch.randperm(len(self.weights), generator=g).tolist()
        rank_indices = shuffled[self.rank : self.num_samples : self.world_size]

        if len(rank_indices) == 0:
            return iter([])

        rank_weights = self.weights[rank_indices]
        rank_weights = rank_weights / rank_weights.sum()

        sampled_local = torch.multinomial(
            rank_weights,
            num_samples=self.num_samples_per_rank,
            replacement=True,
            generator=g,
        )
        return iter([rank_indices[i] for i in sampled_local.tolist()])

    def __len__(self) -> int:
        return self.num_samples_per_rank


class RLDSStateActionDataset:
    """
    PyTorch Dataset：从 RLDS 加载 state 和 action。

    每个样本为长度为 horizon 的滑动窗口。
    - horizon=1: 单步 (state, action)
    - horizon>1: 多步窗口 (state_window, action_window)

    输出模式（互斥）：
      - action_only=True: 仅返回 action_window，shape (horizon, action_dim)
      - state_only=True:  仅返回 state_window，shape (horizon, state_dim)
      - 两者均为 False:   返回 (state_window, action_window)
    """

    def __init__(
        self,
        root_dir: str,
        max_episodes: Optional[int] = None,
        horizon: int = 1,
        preload_cache_path: Optional[str] = None,
        num_parallel_reads: int = 4,
        action_only: bool = False,
        state_only: bool = False,
        debug: bool = False,
        dataset_name: str = None,
    ):
        """
        Args:
            root_dir: RLDS 根目录
            max_episodes: 最多加载的 episode 数，None 表示全部
            horizon: 每个样本的步数（滑动窗口长度）
            preload_cache_path: pickle 缓存路径，加速重复加载
            num_parallel_reads: TFRecord 并行读取数
            action_only: 仅返回 action
            state_only: 仅返回 state
            debug: 调试模式，只加载一个 episode
        """
        if action_only and state_only:
            raise ValueError("action_only 与 state_only 互斥")

        self.horizon = horizon
        self.action_only = action_only
        self.state_only = state_only

        self.episode_states, self.episode_actions = load_rlds_state_action(
            root_dir=root_dir,
            max_episodes=max_episodes,
            preload_cache_path=preload_cache_path,
            num_parallel_reads=num_parallel_reads,
            debug=debug,
            dataset_name=dataset_name,
        )

        # 构建样本索引: (episode_idx, start_idx)
        self.samples: List[Tuple[int, int]] = []
        for ep_idx, states in enumerate(self.episode_states):
            num_samples = len(states) - self.horizon + 1
            for start_idx in range(num_samples):
                self.samples.append((ep_idx, start_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ep_idx, start_idx = self.samples[idx]
        end_idx = start_idx + self.horizon

        if self.action_only:
            return np.asarray(self.episode_actions[ep_idx][start_idx:end_idx], dtype=np.float32)
        if self.state_only:
            return np.asarray(self.episode_states[ep_idx][start_idx:end_idx], dtype=np.float32)

        state_window = np.asarray(self.episode_states[ep_idx][start_idx:end_idx], dtype=np.float32)
        action_window = np.asarray(self.episode_actions[ep_idx][start_idx:end_idx], dtype=np.float32)
        return state_window, action_window


# =============================================================================
# 统一 DataLoader 构造接口
# =============================================================================

def make_rlds_dataloader(
    dataset_specs: List[Tuple[str, float]],
    data_root: str = "data",
    horizon: int = 10,
    batch_size: int = 64,
    action_only: bool = True,
    num_parallel_reads: int = 8,
    max_episodes: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    debug: bool = False,
    # normalize + pad（在 dataset 加载时完成，避免混 stats、统一维度）
    target_action_dim: Optional[int] = None,
    target_horizon: Optional[int] = None,
    stats_path_map: Optional[Dict[str, str]] = None,
    normalizer_method: str = "qq",
    normalizer_method_map: Optional[Dict[str, str]] = None,
    normalizer_config: Optional[dict] = None,
    action_dim_map: Optional[Dict[str, int]] = None,
    horizon_map: Optional[Dict[str, int]] = None,
    action_start_map: Optional[Dict[str, int]] = None,
    # 评估模式：顺序遍历，不使用加权采样
    eval_mode: bool = False,
    # DDP 兼容
    distributed: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    drop_last: bool = False,
) -> Tuple[Optional[Sampler], DataLoader, ConcatDataset]:
    """
    根据数据集规格列表构造加权 DataLoader。

    Args:
        dataset_specs: [(dataset_path, weight), ...]
            例如 [("bridge", 1.0), ("Libero_RLDS/libero_spatial_no_noops", 0.5)]
            - dataset_path: 相对于 data_root 的路径
            - weight: 采样权重，越大该数据集被采到的概率越高
        data_root: 数据根目录，默认 "data"
        horizon: 滑动窗口长度
        batch_size: 批大小
        action_only: 是否只返回 action
        num_parallel_reads: TFRecord 并行读取数
        max_episodes: 每个数据集最多加载的 episode 数，None 表示全部
        num_workers: DataLoader 进程数，0 可避免 CephFS 下的 ConnectionResetError
        pin_memory: 是否 pin memory
        debug: 调试模式，每个数据集只加载一个 episode
        target_action_dim: 输出 action 的 pad 维度，None 表示不 pad
        stats_path_map: dataset_path -> stats_path，每个数据集用各自 stats 做 normalize
        normalizer_method: 默认 normalizer 类型
        normalizer_method_map: dataset_path -> 该数据集用的 normalizer 类型（如 bridge 用 "bridge"）
        normalizer_config: normalizer 额外配置（如 clip, action_dim）
        distributed: 是否 DDP 模式，True 时使用 DistributedWeightedSampler
        rank: rank（distributed 时），None 则从 accelerate.PartialState 自动获取
        world_size: world_size（distributed 时），None 则从 PartialState 自动获取
        drop_last: 是否 drop_last 使各 rank batch 数对齐。DDP 时建议 True。

    Note:
        DDP 模式下需每 epoch 调用 loader.sampler.set_epoch(epoch) 以保证 shuffle 不同。

    Returns:
        ``(sampler, loader, dataset)``
    """
    if distributed:
        try:
            from accelerate import PartialState
            state = PartialState()
            _rank = state.process_index
            _world_size = state.num_processes
        except ImportError:
            _rank = rank if rank is not None else 0
            _world_size = world_size if world_size is not None else 1

    datasets: List[Union[RLDSStateActionDataset, object]] = []
    dataset_weights: List[Tuple[float, int]] = []  # (weight, len) per dataset

    for dataset_path, weight in dataset_specs:
        root_dir = f"{data_root}/{dataset_path}".rstrip("/")
        cache_path = f"{root_dir}/rlds_state_action_cache.pkl"

        # 每个数据集可使用独立的 horizon（滑动窗口长度）
        ds_horizon = horizon_map.get(dataset_path, horizon) if horizon_map else horizon

        ds = RLDSStateActionDataset(
            root_dir=root_dir,
            max_episodes=max_episodes,
            horizon=ds_horizon,
            preload_cache_path=cache_path,
            num_parallel_reads=num_parallel_reads,
            action_only=action_only,
            state_only=False,
            debug=debug,
        )

        # 若指定了 normalize / pad 相关参数，则包装 transform
        need_transform = (
            target_action_dim is not None
            or stats_path_map is not None
            or (horizon_map and ds_horizon != (target_horizon or horizon))
        )
        if need_transform:
            stats_path = stats_path_map.get(dataset_path) if stats_path_map else None
            method = (
                normalizer_method_map.get(dataset_path, normalizer_method)
                if normalizer_method_map
                else normalizer_method
            )
            # 为每个数据集构建独立的 normalizer_config，支持 per-dataset action_dim
            ds_normalizer_config = dict(normalizer_config) if normalizer_config else {}
            if action_dim_map and dataset_path in action_dim_map:
                ds_normalizer_config['action_dim'] = action_dim_map[dataset_path]
            # 仅当该数据集 horizon 与 target 不同时才 pad horizon
            ds_target_horizon = target_horizon if (ds_horizon != (target_horizon or horizon)) else None
            ds_action_start = action_start_map.get(dataset_path, 0) if action_start_map else 0
            ds = wrap_dataset_with_transform(
                dataset=ds,
                stats_path=stats_path,
                target_action_dim=target_action_dim if target_action_dim is not None else 8,
                target_horizon=ds_target_horizon,
                action_only=action_only,
                normalizer_method=method,
                normalizer_config=ds_normalizer_config,
                action_start=ds_action_start,
            )

        datasets.append(ds)
        dataset_weights.append((weight, len(ds)))

    concat_ds = ConcatDataset(datasets)
    num_total = len(concat_ds)

    if eval_mode:
        # 评估模式：顺序遍历所有样本，不加权采样
        sampler = None
        loader = DataLoader(
            concat_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
    else:
        # 向量化构建权重，避免逐样本 extend
        weight_arrays = [np.full(n, w / n, dtype=np.float64) for w, n in dataset_weights]
        weights = torch.from_numpy(np.concatenate(weight_arrays))

        if distributed:
            sampler = DistributedWeightedSampler(
                weights=weights,
                num_samples=num_total,
                rank=_rank,
                world_size=_world_size,
                drop_last=drop_last,
            )
        else:
            sampler = WeightedRandomSampler(
                weights,
                num_samples=num_total,
                replacement=True,
            )

        loader = DataLoader(
            concat_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
        )

    return sampler, loader, concat_ds

# =============================================================================
# 命令行测试
# =============================================================================

if __name__ == "__main__":
    # 使用统一接口构造 DataLoader（含 normalize + pad）
    _, loader, _ = make_rlds_dataloader(
        dataset_specs=[
            ("bridge", 1.0),
            ("Libero_RLDS/libero_spatial_no_noops", 5.0),
            ("Libero_RLDS/libero_goal_no_noops", 5.0),
            ("Libero_RLDS/libero_object_no_noops", 5.0),
            ("Libero_RLDS/libero_10_no_noops", 5.0),
        ],
        data_root="data",
        horizon=10,
        batch_size=64,
        action_only=True,
        target_action_dim=8,
        stats_path_map={
            "bridge": "data/bridge/stats.json",
            "Libero_RLDS/libero_spatial_no_noops": "data/Libero_RLDS/libero_spatial_no_noops/stats.json",
            "Libero_RLDS/libero_goal_no_noops": "data/Libero_RLDS/libero_goal_no_noops/stats.json",
            "Libero_RLDS/libero_object_no_noops": "data/Libero_RLDS/libero_object_no_noops/stats.json",
            "Libero_RLDS/libero_10_no_noops": "data/Libero_RLDS/libero_10_no_noops/stats.json",
        },
        normalizer_method_map={"bridge": "bridge", "Libero_RLDS/libero_spatial_no_noops": "qq", "Libero_RLDS/libero_goal_no_noops": "qq", "Libero_RLDS/libero_object_no_noops": "qq", "Libero_RLDS/libero_10_no_noops": "qq"},
        normalizer_config={"clip": True, "action_dim": 7},
    )
    batch = next(iter(loader))
    print("action:", batch.shape, batch.dtype)  # (64, 10, 8)
