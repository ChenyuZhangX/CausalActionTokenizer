import torch
import numpy as np

from catok.models.catok_ddt.vanilla_utils import load_checkpoint
from catok.training.statistics import build_action_normalizer

from tqdm import tqdm
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import os

import tensorflow_datasets as tfds
import tensorflow as tf

def load_dataset(dataset_dir):
    """Load a single RLDS dataset."""
    # Find the version directory (e.g., 1.0.0)
    version_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not version_dirs:
        print(f"Warning: No version directory found in {dataset_dir}")
        return
    
    version_dir = version_dirs[0]  # Use first version found
    
    # Load dataset using tensorflow_datasets
    builder = tfds.builder_from_directory(str(version_dir))
    ds = builder.as_dataset(split='train')
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds
    
def get_actions_states_per_episode(episode, use_joint_state):
    states_list = []
    actions_list = []
    
    for step in episode['steps']:
        state = step['observation']['joint_state'].numpy()
        action = step['action'].numpy()
        states_list.append(state)
        actions_list.append(action)
    
    states = np.array(states_list, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.float32)
    
    # Delta = next_joint_state - current_joint_state (7D)
    if use_joint_state:
        delta_actions = states[1:]
        # Gripper position from action (last dim of EEF action, 1D)
        gripper_actions = actions[:-1, -1:]
        # Final action: (delta_joint_state, gripper_position) = 8D
        actions = np.concatenate([delta_actions, gripper_actions], axis=-1)
        states = states[:-1]
    
    return actions, states


def visualize_trajs(args):
    model_path = args.model_path
    device = args.device

    model, cfg = load_checkpoint(model_path, device)
    model.eval()

    action_dim = cfg['tokenizer']['basic']['action_dim']
    horizon = cfg['tokenizer']['basic']['horizon']

    if args.one_step:
        output_path = cfg['exp_name'] + '_one_step'
    else:
        output_path = cfg['exp_name'] + f'_{args.steps}'
    os.makedirs(f'outputs/{output_path}', exist_ok=True)

    use_joint_state = cfg['dataset']['dataset_config']['use_joint_state']
    extra_delta = cfg['dataset']['dataset_config']['extra_delta']
    def fetch_func(x):
        x = x[:, :-1]
        # padding to action_dim
        x = np.pad(x, ((0, 0), (0, action_dim - x.shape[-1])), mode='constant')
        return x
    normalizer_cfg = cfg['normalizer']
    normalizer = build_action_normalizer(
        method=normalizer_cfg['type'],
        normalizer_config=normalizer_cfg['config']
    )
    
    root_dir = Path(args.data_root)
    dataset_dirs = [d for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    for dataset_dir in dataset_dirs:
        ds = load_dataset(dataset_dir)
        for idx, episode in tqdm(enumerate(ds), desc=f"Loading {dataset_dir.name}"):
            i = dataset_dir.name + str(idx)
            actions, states = get_actions_states_per_episode(episode, use_joint_state)

            with torch.no_grad():
                actions = fetch_func(actions)

                # Ensure states and actions have the same length
                min_len = min(len(states), len(actions))
                # states = states[:min_len]
                actions = actions[:min_len]

                actions_batch = torch.from_numpy(actions).to(args.device)   # (episode_len, d)
                states_batch = torch.from_numpy(states).to(args.device)     # (episode_len, d)

                # get delta action chunks
                actions_batch = actions_batch.unfold(0, horizon, 1).clone().contiguous().permute(0, 2, 1)   # (episode_len - horizon + 1, horizon, d)
                if extra_delta:
                    actions_batch[..., :-1] = actions_batch[..., :-1] - states_batch[:-horizon+1, None]
                actions_batch = normalizer.normalize(actions_batch.cpu().detach().numpy())
                actions_batch = torch.from_numpy(actions_batch).to(
                    device=args.device,
                    dtype=torch.float32
                )

                # reconstruction absolute action chunk
                pred_actions = model.reconstruct(actions_batch, steps=args.steps, one_step=args.one_step)
                
                # pred_actions = normalizer.denormalize(pred_actions.cpu().detach().numpy())
                # pred_actions = torch.from_numpy(pred_actions).to(
                #     device=args.device,
                #     dtype=torch.float32
                # )
                # pred_actions[..., :-1] = pred_actions[..., :-1] + states_batch[:-horizon+1, None]
            
            # plot actions in a figure
            episode_len, action_dim = actions.shape
            state_dim = states.shape[1]
            steps = np.arange(episode_len)
            steps_batch = np.lib.stride_tricks.sliding_window_view(steps, window_shape=horizon)
            
            # Plot combined figure: actions and states side by side
            fig, axes = plt.subplots(nrows=max(action_dim, state_dim), ncols=1, figsize=(8, 12), sharex='col')
            # Plot actions in left column
            for j in range(action_dim):
                for k in range(steps_batch.shape[0]//horizon):
                    axes[j].plot(
                        steps_batch[k*horizon],
                        pred_actions[k*horizon, :, j].cpu().numpy().reshape(-1),
                        color='r', linewidth=1.5, alpha=0.7
                    )
                    axes[j].plot(
                        steps_batch[k*horizon],
                        actions_batch[k*horizon, :, j].cpu().numpy().reshape(-1),
                        color='b', linewidth=1.5, alpha=0.7
                    )
                axes[j].set_ylabel(f'action_{j}')
                axes[j].legend(loc='upper right')
                axes[j].grid(True, alpha=0.3)
                axes[j].set_title('Actions')
                axes[j].set_ylim((-1, 1))
            # Hide extra action subplots if state_dim > action_dim
            for j in range(action_dim, max(action_dim, state_dim)):
                axes[j, 0].axis('off')
            
            plt.tight_layout()
            plt.xlabel('Time Steps')
            plt.savefig(f'outputs/{output_path}/{i}.png', dpi=150)
            plt.close(fig)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Causal Action Tokenizer Reconstruction")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--steps', type=int, default=20, help='Diffusion steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the evaluation on')
    parser.add_argument('--data-root', default='data/Libero_RLDS', help='LIBERO RLDS root directory')
    parser.add_argument('--one_step', action='store_true', help='Whether to use one-step reconstruction')
    args = parser.parse_args()
    
    visualize_trajs(args)
