"""RLDS state/action loading utilities.

从 RLDS TFRecord 格式（如 LIBERO）中加载 state 和 action，不加载图像。

主要功能：
  - load_rlds_state_action: 加载 RLDS 目录，返回 episode 列表
  - 支持 pickle 缓存加速重复加载
  - 支持并行 TFRecord 读取
"""
import random
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tqdm

try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
except ImportError:
    tfds = None
    tf = None


# =============================================================================
# 公开 API
# =============================================================================

def load_rlds_state_action(
    root_dir: str,
    max_episodes: Optional[int] = None,
    preload_cache_path: Optional[str] = None,
    num_parallel_reads: int = 4,
    debug: bool = False,
    dataset_name: str = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    从 RLDS 目录加载 state 和 action，按 episode 返回。

    支持标准 LIBERO/Bridge RLDS TFRecord 格式和单文件 pickle 缓存。

    Args:
        root_dir: 数据根目录
        max_episodes: 最多加载的 episode 数，None 表示全部
        preload_cache_path: pickle 缓存路径（单文件格式 {"episode_states":..., "episode_actions":...}）
        num_parallel_reads: TFRecord 并行读取数
        debug: 为 True 时只加载第一个 episode

    Returns:
        episode_states: List[np.ndarray]，每个 shape (T, state_dim)
        episode_actions: List[np.ndarray]，每个 shape (T, action_dim)
    """
    root = Path(root_dir)
    # 1. 尝试从单文件缓存加载
    if preload_cache_path:
        cached = _load_from_cache(preload_cache_path, max_episodes)
        if cached is not None:
            return cached

    # 2. 从 RLDS 加载
    if tfds is None:
        raise ImportError(
            "需要 tensorflow_datasets。安装: pip install tensorflow_datasets"
        )

    episode_states: List[np.ndarray] = []
    episode_actions: List[np.ndarray] = []

    root = Path(root_dir)
    dataset_dirs = [
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue
        _load_single_dataset(
            dataset_dir=dataset_dir,
            episode_states=episode_states,
            episode_actions=episode_actions,
            num_parallel_reads=num_parallel_reads,
            debug=debug,
            dataset_name=dataset_name,
        )
        if debug and len(episode_states) > 0:
            break

    # 3. 按 max_episodes 子采样
    if max_episodes is not None and len(episode_states) > max_episodes:
        indices = random.sample(range(len(episode_states)), max_episodes)
        episode_states = [episode_states[i] for i in indices]
        episode_actions = [episode_actions[i] for i in indices]

    # 4. 写入缓存
    if preload_cache_path:
        _save_to_cache(preload_cache_path, episode_states, episode_actions)

    return episode_states, episode_actions


# =============================================================================
# 内部实现
# =============================================================================

def _load_from_cache(
    cache_path: str,
    max_episodes: Optional[int],
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """从 pickle 缓存加载，若不存在返回 None。"""
    path = Path(cache_path)
    if not path.exists():
        return None

    with open(path, "rb") as f:
        data = pickle.load(f)

    episode_states = data["episode_states"]
    episode_actions = data["episode_actions"]

    if max_episodes is not None and len(episode_states) > max_episodes:
        indices = random.sample(range(len(episode_states)), max_episodes)
        episode_states = [episode_states[i] for i in indices]
        episode_actions = [episode_actions[i] for i in indices]

    return episode_states, episode_actions


def _save_to_cache(
    cache_path: str,
    episode_states: List[np.ndarray],
    episode_actions: List[np.ndarray],
) -> None:
    """将 episode 数据写入 pickle 缓存。"""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {"episode_states": episode_states, "episode_actions": episode_actions},
            f,
        )
    print(f"已保存缓存: {path}")


def _load_single_dataset(
    dataset_dir: Path,
    episode_states: List[np.ndarray],
    episode_actions: List[np.ndarray],
    num_parallel_reads: int = 4,
    debug: bool = False,
    dataset_name: str = None
) -> None:
    """
    从单个 RLDS 版本目录加载，结果追加到 episode_states / episode_actions。
    """
    builder = tfds.builder_from_directory(str(dataset_dir))
    read_config = tfds.ReadConfig(
        interleave_cycle_length=num_parallel_reads,
    )
    ds = builder.as_dataset(split="train", read_config=read_config)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    for episode in tqdm.tqdm(ds, desc=f"Loading {dataset_dir.name}"):
        states_list: List[np.ndarray] = []
        actions_list: List[np.ndarray] = []

        for step in episode["steps"]:
            if dataset_name == "kuka":
                state = np.concatenate([step["observation"]["clip_function_input/base_pose_tool_reached"].numpy(), step["observation"]["gripper_closed"].numpy()])

                gripper_value = step["action"]["gripper_closedness_action"].numpy()
                assert -1 - 1e-3 < gripper_value < 1 + 1e-3, f"Gripper value out of range: {gripper_value}"
                action = np.concatenate([
                    step["action"]["world_vector"].numpy(),
                    step["action"]["rotation_delta"].numpy(),
                    step["action"]["gripper_closedness_action"].numpy()
                ])
            else:
                state = step["observation"]["state"].numpy()
                action = step["action"].numpy()
            states_list.append(state)
            actions_list.append(action)

        if len(states_list) == 0:
            continue

        episode_states.append(np.array(states_list, dtype=np.float32))
        episode_actions.append(np.array(actions_list, dtype=np.float32))

        if debug:
            break
