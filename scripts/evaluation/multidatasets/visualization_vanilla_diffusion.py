"""
Multi-dataset Visualization for VanillaVQDiffusion

Per-episode visualization: load episodes from multiple RLDS datasets,
reconstruct action chunks via tokenizer, and plot original vs reconstructed.

Dataset specs (stats_path, normalizer_method, action_dim, horizon) are read
from the checkpoint config (data.datasets), so no hardcoded constants are needed.

model.reconstruct() handles padding/cropping automatically, so actions are passed
at their native (ds_horizon, ds_action_dim) and returned at the same shape.

Usage:
    python scripts/evaluation/multidatasets/visualization_vanilla_diffusion.py \\
        -m path/to/ckpt.pth \\
        --steps 20 \\
        --device cuda \\
        --max_episodes_per_dataset 3
"""

import torch
import numpy as np

from catok.models.catok_ddt.vanilla_utils import load_checkpoint
from catok.training.rlds_dataset import RLDSStateActionDataset
from catok.training.batch_transform import build_action_transform

from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt


def _parse_dataset_specs_from_cfg(cfg):
    """从 cfg['data']['datasets'] 解析 per-dataset 参数。"""
    datasets_cfg = cfg.get('data', {}).get('datasets', {})
    default_method = cfg.get('normalizer', {}).get('type', 'qq')
    default_action_dim = cfg['tokenizer']['basic']['action_dim']
    default_horizon = cfg['tokenizer']['basic']['horizon']

    specs = []
    for ds_path, ds_cfg in datasets_cfg.items():
        specs.append({
            'dataset_path': ds_path,
            'stats_path': ds_cfg.get('stats_path'),
            'normalizer_method': ds_cfg.get('normalizer_method', default_method),
            'normalizer_config': cfg.get('normalizer', {}).get('config', {}),
            'action_dim': ds_cfg.get('action_dim', default_action_dim),
            'horizon': ds_cfg.get('horizon', default_horizon),
        })
    return specs


def plot_episode(
    actions_norm, pred_actions, dataset_name, ep_idx,
    horizon, output_dir, action_dim, stride=1,
):
    """Plot original vs reconstructed action chunks for one episode.

    Args:
        actions_norm: (N, horizon, action_dim) tensor — normalized action chunks
        pred_actions:  (N, horizon, action_dim) tensor — reconstructed action chunks
        stride: step between consecutive chunks (1 = sliding window, horizon = equal chunks)
    """
    num_chunks = actions_norm.shape[0]
    episode_len = (num_chunks - 1) * stride + horizon

    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 2.5 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]

    alpha = 0.6 if stride >= horizon else 0.3
    for j in range(action_dim):
        for k in range(num_chunks):
            x = np.arange(k * stride, k * stride + horizon)
            axes[j].plot(
                x,
                pred_actions[k, :, j].cpu().numpy(),
                color='r', linewidth=0.8, alpha=alpha,
                label='Reconstructed' if k == 0 else None,
            )
            axes[j].plot(
                x,
                actions_norm[k, :, j].cpu().numpy(),
                color='b', linewidth=0.8, alpha=alpha,
                label='Original' if k == 0 else None,
            )
        axes[j].set_ylabel(f'dim {j}')
        axes[j].grid(True, alpha=0.3)
        axes[j].set_ylim((-1.2, 1.2))
        if j == 0:
            axes[j].legend(loc='upper right', fontsize=8)

    mode_tag = 'equal_chunks' if stride >= horizon else f'stride={stride}'
    axes[-1].set_xlabel('Time Steps')
    fig.suptitle(f'{dataset_name}  ep={ep_idx}  T={episode_len}  [{mode_tag}]', fontsize=12)
    plt.tight_layout()

    safe_name = dataset_name.replace('/', '_')
    fname = f'{output_dir}/{safe_name}_ep{ep_idx}.png'
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def visualize_trajs(args):
    model_path = args.model_path
    device = args.device

    model, cfg = load_checkpoint(model_path, device)
    model.eval()

    data_cfg = cfg['data']
    dataset_specs = _parse_dataset_specs_from_cfg(cfg)

    if args.one_step:
        output_dir = f'outputs/{cfg["exp_name"]}_vis_one_step'
    else:
        output_dir = f'outputs/{cfg["exp_name"]}_vis_{args.steps}steps'
    os.makedirs(output_dir, exist_ok=True)

    for spec in dataset_specs:
        dataset_path = spec['dataset_path']
        ds_horizon = spec['horizon']
        ds_action_dim = spec['action_dim']

        root_dir = f"{data_cfg['root_dir']}/{dataset_path}"
        cache_path = f"{root_dir}/rlds_state_action_cache.pkl"

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_path}  (horizon={ds_horizon}, action_dim={ds_action_dim})")
        print(f"{'='*60}")

        try:
            ds = RLDSStateActionDataset(
                root_dir=root_dir,
                horizon=ds_horizon,
                preload_cache_path=cache_path,
                action_only=True,
                debug=False,
            )
        except Exception as e:
            print(f"  Skipping {dataset_path}: {e}")
            continue

        # Per-dataset normalizer (only normalize, no padding — model handles padding)
        normalizer = build_action_transform(
            stats_path=spec['stats_path'],
            normalizer_method=spec['normalizer_method'],
            normalizer_config=spec['normalizer_config'],
        )

        num_episodes = len(ds.episode_actions)
        max_ep = min(args.max_episodes_per_dataset, num_episodes)
        ep_indices = np.linspace(0, num_episodes - 1, max_ep, dtype=int)

        print(f"  Total episodes: {num_episodes}, visualizing: {len(ep_indices)}")

        stride = ds_horizon if args.equal_chunks else 1

        for ep_idx in tqdm(ep_indices, desc=f"  {dataset_path}"):
            actions = ds.episode_actions[ep_idx]  # (T, ds_action_dim)

            actions_t = torch.from_numpy(actions).float()
            if actions_t.shape[0] < ds_horizon:
                continue

            # Chunk episode: stride=1 → sliding window; stride=ds_horizon → equal non-overlapping chunks
            actions_batch = actions_t.unfold(0, ds_horizon, stride).permute(0, 2, 1).contiguous()

            # Normalize at native ds_action_dim (no manual padding)
            if normalizer is not None:
                actions_norm_np = normalizer.normalize(actions_batch.numpy())
            else:
                actions_norm_np = actions_batch.numpy()
            actions_norm = torch.from_numpy(actions_norm_np).to(device=device, dtype=torch.float32)

            # reconstruct: auto-pads to model dims, returns cropped to (N, ds_horizon, ds_action_dim)
            with torch.no_grad():
                pred_actions = model.reconstruct(actions_norm, steps=args.steps, one_step=args.one_step)

            plot_episode(
                actions_norm, pred_actions,
                dataset_path, ep_idx,
                ds_horizon, output_dir, ds_action_dim,
                stride=stride,
            )

        print(f"  Saved to {output_dir}/")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-dataset VanillaVQDiffusion Visualization")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--steps', type=int, default=20, help='Diffusion sampling steps')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--one_step', action='store_true', help='Use one-step reconstruction')
    parser.add_argument('--max_episodes_per_dataset', type=int, default=3,
                        help='Max episodes to visualize per dataset')
    parser.add_argument('--equal_chunks', action='store_true',
                        help='Divide episode into equal non-overlapping chunks (stride=horizon) '
                             'instead of sliding window (stride=1)')
    args = parser.parse_args()

    visualize_trajs(args)
