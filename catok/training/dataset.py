"""Single-dataset LIBERO RLDS loader used by the public README commands."""

import pickle
import random
from pathlib import Path

import numpy as np
import tqdm
from torch.utils.data import Dataset

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ImportError:
    tf = None
    tfds = None


def make_dataset(dataset_name, dataset_config):
    if dataset_name != "libero_rlds":
        raise ValueError(f"Unknown public dataset: {dataset_name}")
    return RLDSStateActionDataset(**dataset_config)


class RLDSStateActionDataset(Dataset):
    """Load action/state windows from one or more LIBERO RLDS datasets."""

    def __init__(
        self,
        root_dir,
        horizon=8,
        extra_delta=True,
        max_episodes=None,
        action_only=False,
        state_only=False,
        absolute_gripper=True,
        action_space="eef",
        use_joint_state=False,
        dataset_names=None,
        debug=False,
        preload=False,
    ):
        if action_only and state_only:
            raise ValueError("action_only and state_only are mutually exclusive")
        if tf is None or tfds is None:
            raise ImportError("LIBERO RLDS loading requires tensorflow and tensorflow-datasets")

        self.root_dir = Path(root_dir)
        self.horizon = horizon
        self.extra_delta = extra_delta
        self.action_only = action_only
        self.state_only = state_only
        self.absolute_gripper = absolute_gripper
        self.action_space = action_space
        if action_space == "eef":
            self.use_joint_state = False
        elif action_space == "joint":
            self.use_joint_state = True
        else:
            raise ValueError(f"Invalid action space: {action_space}")
        self.debug = debug

        # TensorFlow is used only for input; leave accelerator ownership to PyTorch.
        tf.config.set_visible_devices([], "GPU")

        if preload:
            suffix = "" if use_joint_state and extra_delta else "_EEF_H20"
            with open(self.root_dir / f"libero_states{suffix}.pkl", "rb") as file:
                self.episode_states = pickle.load(file)
            with open(self.root_dir / f"libero_actions{suffix}.pkl", "rb") as file:
                self.episode_actions = pickle.load(file)
            with open(self.root_dir / f"libero_samples{suffix}.pkl", "rb") as file:
                self.samples = pickle.load(file)
        else:
            dataset_dirs = (
                [path for path in self.root_dir.iterdir() if path.is_dir() and not path.name.startswith(".")]
                if dataset_names is None
                else [self.root_dir / name for name in dataset_names]
            )
            self.episode_states = []
            self.episode_actions = []
            for dataset_dir in dataset_dirs:
                self._load_dataset(dataset_dir)
                if debug and self.episode_states:
                    break

            if max_episodes is not None and len(self.episode_states) > max_episodes:
                indices = random.sample(range(len(self.episode_states)), max_episodes)
                self.episode_states = [self.episode_states[index] for index in indices]
                self.episode_actions = [self.episode_actions[index] for index in indices]

            self.samples = [
                (episode_index, start_index)
                for episode_index, states in enumerate(self.episode_states)
                for start_index in range(len(states) - self.horizon + 1)
            ]

    def _load_dataset(self, dataset_dir):
        version_dirs = [
            path for path in dataset_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
        ]
        if not version_dirs:
            return

        builder = tfds.builder_from_directory(str(version_dirs[0]))
        episodes = builder.as_dataset(split="train").prefetch(tf.data.AUTOTUNE)
        for episode in tqdm.tqdm(episodes, desc=f"Loading {dataset_dir.name}"):
            states, actions = [], []
            for step in episode["steps"]:
                if self.action_space == "eef":
                    state = step["observation"]["state"].numpy()[:7]
                else:
                    state = step["observation"]["joint_state"].numpy()
                states.append(state)
                actions.append(step["action"].numpy())

            if len(states) < self.horizon:
                continue
            states = np.asarray(states, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.float32)
            if self.use_joint_state:
                actions = np.concatenate([states[1:], actions[:-1, -1:]], axis=-1)
                states = states[:-1]
            if len(states) >= self.horizon:
                self.episode_states.append(states)
                self.episode_actions.append(actions)
                if self.debug:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode_index, start_index = self.samples[index]
        states = self.episode_states[episode_index]
        actions = self.episode_actions[episode_index]
        state_window = states[start_index : start_index + self.horizon].copy()
        action_window = actions[start_index : start_index + self.horizon].copy()

        if self.extra_delta and self.use_joint_state:
            dimensions = min(action_window.shape[1] - 1, state_window.shape[1])
            action_window[:, :-1] -= state_window[:1, :dimensions]

        if self.action_only:
            return action_window
        if self.state_only:
            return state_window
        return state_window, action_window
