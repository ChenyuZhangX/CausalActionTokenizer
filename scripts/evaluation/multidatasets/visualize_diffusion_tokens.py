"""
Two visualization utilities for VanillaVQDiffusion:

1. Diffusion process with incremental tokens (--mode tokens):
   For a given action chunk, reconstruct using 1, 2, ..., num_tokens
   encoder tokens and visualize how the prediction improves as more
   tokens are used. Each subplot corresponds to one action dimension;
   the ground truth is shown in black, and reconstructions are colored
   from light to dark (viridis colormap) as the number of tokens increases.

2. Encoder embedding distribution (--mode embedding):
   After encoding a batch of action chunks, collect the num_tokens
   per-position embeddings (after VQ if applicable), reduce to 2-D
   with t-SNE, and plot all token positions on the same scatter plot
   with different colors.

Usage:
    # Run both visualizations
    python scripts/evaluation/multidatasets/visualize_diffusion_tokens.py \
        -m path/to/ckpt.pth \
        --steps 20 \
        --device cuda \
        --max_episodes_per_dataset 2 \
        --mode both

    # Only incremental-token reconstruction
    python scripts/evaluation/multidatasets/visualize_diffusion_tokens.py \
        -m path/to/ckpt.pth \
        --steps 20 \
        --device cuda \
        --mode tokens

    # Only encoder embedding t-SNE
    python scripts/evaluation/multidatasets/visualize_diffusion_tokens.py \
        -m path/to/ckpt.pth \
        --device cuda \
        --mode embedding \
        --max_tsne_samples 3000 \
        --tsne_perplexity 50

Arguments:
    -m, --model_path                Path to the model checkpoint (.pth)
    --steps                         Number of diffusion sampling steps (default: 20)
    --device                        Device to run on (default: cuda)
    --max_episodes_per_dataset      Max episodes to visualize per dataset (default: 2)
    --mode {tokens,embedding,both}  Which visualization to run (default: both)
    --max_tsne_samples              Max samples fed to t-SNE to keep runtime
                                    tractable (default: 2000)
    --tsne_perplexity               t-SNE perplexity hyperparameter (default: 30)

Outputs are saved to  outputs/<exp_name>_vis_tokens/
"""

import torch
import torch.nn.functional as F
import numpy as np

from catok.models.catok_ddt.vanilla_utils import load_checkpoint, get_mask, construct_dataloader
from catok.training.rlds_dataset import RLDSStateActionDataset
from catok.training.batch_transform import build_action_transform, normalize_and_pad_action

from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE

def _parse_dataset_specs_from_cfg(cfg):
    """从 cfg['data']['datasets'] 解析 per-dataset 信息，用于逐数据集可视化。

    Returns:
        dataset_specs: [(dataset_path, stats_path), ...]
        normalizer_method_map: {dataset_path: method}
        action_dim_map: {dataset_path: action_dim}
        horizon_map: {dataset_path: horizon}
    """
    datasets_cfg = cfg.get('data', {}).get('datasets', {})
    default_method = cfg.get('normalizer', {}).get('type', 'qq')
    dataset_specs = []
    normalizer_method_map = {}
    action_dim_map = {}
    horizon_map = {}
    for ds_path, ds_cfg in datasets_cfg.items():
        stats_path = ds_cfg.get('stats_path', None)
        dataset_specs.append((ds_path, stats_path))
        normalizer_method_map[ds_path] = ds_cfg.get('normalizer_method', default_method)
        if 'action_dim' in ds_cfg:
            action_dim_map[ds_path] = ds_cfg['action_dim']
        if 'horizon' in ds_cfg:
            horizon_map[ds_path] = ds_cfg['horizon']
    return dataset_specs, normalizer_method_map, action_dim_map, horizon_map


# ---------------------------------------------------------------------------
# Helper: sample with a fixed token mask (override the causal mask)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_with_k_tokens(model, shape, encoder_hidden_states, k, steps=20):
    """Run diffusion sampling using only the first k encoder tokens.

    We override the mask so that only the first k tokens are visible at
    every diffusion timestep (ignoring the normal causal schedule).
    """
    decoder = model.decoder
    flow = model.flow
    decoder.eval()

    b = shape[0]
    device = encoder_hidden_states.device
    num_tokens = encoder_hidden_states.shape[1]

    # Fixed mask: first k tokens visible, rest masked
    mask = torch.zeros(b, num_tokens, dtype=torch.bool, device=device)
    mask[:, :k] = True

    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    time_steps = torch.linspace(0, 1, steps + 1, device=device)[:-1]

    for t in time_steps:
        t_batch = torch.ones(b, device=device) * t
        v_pred, _ = decoder(
            x, t_batch,
            encoder_hidden_states=encoder_hidden_states,
            context_see_xt=flow.context_see_xt,
            mask=mask,
        )
        x = x + v_pred * dt

    decoder.train()
    return x


# ---------------------------------------------------------------------------
# Feature 1: Visualize reconstruction with 1..num_tokens tokens
# ---------------------------------------------------------------------------
def plot_incremental_tokens(
    actions_norm, model, dataset_name, ep_idx,
    output_dir, steps, action_dim, chunk_idx=0,
):
    """For a single action chunk, reconstruct with 1..num_tokens tokens and plot."""
    # Pick one chunk
    chunk = actions_norm[chunk_idx:chunk_idx+1]  # (1, horizon, action_dim)
    horizon = chunk.shape[1]

    z = model.encoder(chunk)
    if model.vq_mode is not None:
        _, z_q, _, _ = model.vq(z)
    else:
        z_q = z

    num_tokens = z_q.shape[1]
    shape = chunk.shape  # (1, horizon, action_dim)

    # Collect predictions for k = 1 .. num_tokens
    all_preds = []
    for k in range(1, num_tokens + 1):
        pred = sample_with_k_tokens(model, shape, z_q, k, steps=steps)
        all_preds.append(pred.cpu())

    # Also add full reconstruction (all tokens with causal mask = normal)
    # all_preds[-1] is already the full-token case

    chunk_np = chunk.cpu().numpy()[0]  # (horizon, action_dim)

    # Plot: rows = action dims, cols = 1
    # Each subplot shows the GT and overlaid reconstructions colored by k
    fig, axes = plt.subplots(
        nrows=action_dim, ncols=1,
        figsize=(12, 2.5 * action_dim), sharex=True,
    )
    if action_dim == 1:
        axes = [axes]

    # Use a sequential colormap: light -> dark maps to fewer -> more tokens
    cmap = cm.get_cmap('Reds', num_tokens + 2)  # +2 to avoid pure white at low end
    x_axis = np.arange(horizon)

    for j in range(action_dim):
        ax = axes[j]
        # Ground truth
        ax.plot(x_axis, chunk_np[:, j], 'k-', linewidth=2, label='GT')
        # Predictions: color intensity grows with token count
        for k_idx, pred in enumerate(all_preds):
            k = k_idx + 1
            # Map k to [0.2, 1.0] range so lightest color is still visible
            color = cmap(0.2 + 0.8 * k_idx / max(num_tokens - 1, 1))
            ax.plot(
                x_axis, pred[0, :, j].numpy(),
                color=color, linewidth=1.2,
            )
        ax.set_ylabel(f'dim {j}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim((-1.5, 1.5))

    axes[-1].set_xlabel('Time Steps')

    # Add a colorbar instead of a per-line legend (much cleaner for many tokens)
    norm = plt.Normalize(vmin=1, vmax=num_tokens)
    sm = cm.ScalarMappable(cmap=cm.get_cmap('Reds'), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location='right', shrink=0.6, pad=0.02)
    cbar.set_label('Number of tokens', fontsize=10)
    # Add GT marker in first subplot legend
    axes[0].legend(loc='upper right', fontsize=8)

    fig.suptitle(
        f'{dataset_name}  ep={ep_idx}  chunk={chunk_idx}\n'
        f'Diffusion reconstruction with 1..{num_tokens} tokens ({steps} steps)',
        fontsize=11,
    )
    plt.tight_layout()

    safe_name = dataset_name.replace('/', '_')
    fname = os.path.join(output_dir, f'{safe_name}_ep{ep_idx}_chunk{chunk_idx}_incremental_tokens.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Feature 2: Visualize encoder embedding distribution (t-SNE)
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_embeddings(model, actions_norm, max_samples=2000):
    """Encode action chunks and return embeddings per token position.

    Returns:
        embeddings: (N, num_tokens, d_model) numpy array
    """
    all_z = []
    bs = 256
    for i in range(0, len(actions_norm), bs):
        batch = actions_norm[i:i+bs]
        z = model.encoder(batch)
        if model.vq_mode is not None:
            _, z_q, _, _ = model.vq(z)
        else:
            z_q = z
        all_z.append(z_q.cpu())
    all_z = torch.cat(all_z, dim=0).numpy()
    if all_z.shape[0] > max_samples:
        idx = np.random.choice(all_z.shape[0], max_samples, replace=False)
        all_z = all_z[idx]
    return all_z


def plot_embedding_distribution(
    embeddings, dataset_name, output_dir, perplexity=30,
):
    """Visualize num_tokens embedding classes on one t-SNE plot.

    Args:
        embeddings: (N, num_tokens, d_model) numpy array
    """
    N, num_tokens, d_model = embeddings.shape

    # Flatten to (N * num_tokens, d_model) with labels
    flat = embeddings.reshape(-1, d_model)  # (N*num_tokens, d_model)
    labels = np.repeat(np.arange(num_tokens), N)  # token position labels
    # Note: repeat vs tile — we want [0,0,...,1,1,...,K-1,K-1,...]
    # Actually embeddings is (N, num_tokens, d), reshape(-1, d) gives
    # [sample0_token0, sample0_token1, ..., sample0_tokenK, sample1_token0, ...]
    # So labels should be tiled:
    labels = np.tile(np.arange(num_tokens), N)

    # t-SNE
    perp = min(perplexity, len(flat) // 4)
    tsne_kwargs = dict(n_components=2, perplexity=max(perp, 5), random_state=42)
    try:
        # sklearn>=1.5 uses max_iter
        tsne = TSNE(max_iter=1000, **tsne_kwargs)
    except TypeError:
        # older sklearn uses n_iter
        tsne = TSNE(n_iter=1000, **tsne_kwargs)
    coords = tsne.fit_transform(flat)  # (N*num_tokens, 2)

    safe_name = dataset_name.replace('/', '_')
    fname = os.path.join(output_dir, f'{safe_name}_embedding_tsne.png')
    vivid_fname = os.path.join(output_dir, f'{safe_name}_embedding_tsne_vivid.png')

    def _save_scatter(colors, save_path, title_suffix=""):
        from matplotlib.colors import ListedColormap
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = ListedColormap(colors)

        for tok_idx in range(num_tokens):
            mask = labels == tok_idx
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[colors[tok_idx]], s=8, alpha=0.95,
            )

        norm = plt.Normalize(vmin=0, vmax=num_tokens - 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Token index', fontsize=10)
        cbar.set_ticks(np.arange(num_tokens))

        ax.set_title(f'{dataset_name} — Encoder Embedding t-SNE{title_suffix} (N={N}, tokens={num_tokens})')
        ax.set_xlabel('t-SNE dim 0')
        ax.set_ylabel('t-SNE dim 1')
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    default_colors = cm.get_cmap('coolwarm')(np.linspace(0.0, 1.0, num_tokens))
    vivid_colors = cm.get_cmap('nipy_spectral')(np.linspace(0.02, 0.98, num_tokens))
    _save_scatter(default_colors, fname, "")
    _save_scatter(vivid_colors, vivid_fname, " (vivid)")
    print(f"    Saved vivid version: {vivid_fname}")
    return fname


def plot_embedding_distribution_umap(
    embeddings, dataset_name, output_dir, n_neighbors=15, min_dist=0.1,
):
    """Visualize num_tokens embedding classes on one UMAP plot.

    Args:
        embeddings: (N, num_tokens, d_model) numpy array
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
    """
    import umap

    N, num_tokens, d_model = embeddings.shape

    flat = embeddings.reshape(-1, d_model)
    labels = np.tile(np.arange(num_tokens), N)

    # UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(flat)

    safe_name = dataset_name.replace('/', '_')
    fname = os.path.join(output_dir, f'{safe_name}_embedding_umap.png')
    vivid_fname = os.path.join(output_dir, f'{safe_name}_embedding_umap_vivid.png')

    def _save_scatter(colors, save_path, title_suffix=""):
        from matplotlib.colors import ListedColormap
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = ListedColormap(colors)

        for tok_idx in range(num_tokens):
            mask = labels == tok_idx
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[colors[tok_idx]], s=8, alpha=0.95,
            )

        norm = plt.Normalize(vmin=0, vmax=num_tokens - 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Token index', fontsize=10)
        cbar.set_ticks(np.arange(num_tokens))

        ax.set_title(f'{dataset_name} — Encoder Embedding UMAP{title_suffix} (N={N}, tokens={num_tokens})')
        ax.set_xlabel('UMAP dim 0')
        ax.set_ylabel('UMAP dim 1')
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    default_colors = cm.get_cmap('coolwarm')(np.linspace(0.0, 1.0, num_tokens))
    vivid_colors = cm.get_cmap('nipy_spectral')(np.linspace(0.02, 0.98, num_tokens))
    _save_scatter(default_colors, fname, "")
    _save_scatter(vivid_colors, vivid_fname, " (vivid)")
    print(f"    Saved vivid version: {vivid_fname}")
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    model_path = args.model_path
    device = args.device

    model, cfg = load_checkpoint(model_path, device)
    model.eval()

    action_dim = cfg['tokenizer']['basic']['action_dim']
    horizon = cfg['tokenizer']['basic']['horizon']
    data_cfg = cfg['data']
    normalizer_config = cfg.get('normalizer', {}).get('config', {})

    dataset_specs, normalizer_method_map, action_dim_map, horizon_map = _parse_dataset_specs_from_cfg(cfg)

    output_dir = f'outputs/vis_tokens/{cfg["exp_name"]}'
    os.makedirs(output_dir, exist_ok=True)

    for dataset_path, stats_path in dataset_specs:
        root_dir = f"{data_cfg['root_dir']}/{dataset_path}"
        cache_path = f"{root_dir}/rlds_state_action_cache.pkl"
        ds_horizon = horizon_map.get(dataset_path, horizon)

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_path}")
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

        normalizer_method = normalizer_method_map.get(dataset_path, "qq")
        ds_normalizer_config = dict(normalizer_config) if normalizer_config else {}
        if dataset_path in action_dim_map:
            ds_normalizer_config['action_dim'] = action_dim_map[dataset_path]
        normalizer = build_action_transform(
            stats_path=stats_path,
            normalizer_method=normalizer_method,
            normalizer_config=ds_normalizer_config,
        )

        num_episodes = len(ds.episode_actions)
        max_ep = min(args.max_episodes_per_dataset, num_episodes)
        ep_indices = np.linspace(0, num_episodes - 1, max_ep, dtype=int)

        # ---- Feature 1: incremental token reconstruction ----
        if args.mode in ('tokens', 'both', 'both_umap'):
            print(f"  [1/2] Incremental token visualization ...")
            for ep_idx in tqdm(ep_indices, desc="  incr-tokens"):
                actions = ds.episode_actions[ep_idx]
                actions_t = torch.from_numpy(actions).float()
                if actions_t.shape[0] < horizon:
                    continue
                actions_batch = actions_t.unfold(0, horizon, 1).permute(0, 2, 1).contiguous()
                actions_norm = normalize_and_pad_action(actions_batch.numpy(), normalizer, action_dim)
                actions_norm = torch.from_numpy(actions_norm).to(device=device, dtype=torch.float32)

                # Visualize for a few chunks spread across the episode
                num_chunks = actions_norm.shape[0]
                chunk_indices = np.linspace(0, num_chunks - 1, min(3, num_chunks), dtype=int)
                for ci in chunk_indices:
                    fname = plot_incremental_tokens(
                        actions_norm, model, dataset_path, ep_idx,
                        output_dir, args.steps, action_dim, chunk_idx=ci,
                    )
                    print(f"    Saved: {fname}")

    print(f"  Saved incremental token plots to: {output_dir}")

    # ---- Feature 2: embedding distribution (using dataloader) ----
    if args.mode in ('embedding', 'embedding_umap', 'both', 'both_umap'):
        print(f"\n{'='*60}")
        print("Embedding distribution visualization (dataloader sampling)")
        print(f"{'='*60}")

        _, loader, _ = construct_dataloader(
            cfg,
            batch_size=min(args.max_embed_samples, 4096),
            num_workers=16,
            eval_mode=True,
        )

        # Sample action chunks from dataloader
        all_chunks = []
        collected = 0
        for action_chunks in tqdm(loader, desc="  sampling embeddings"):
            all_chunks.append(action_chunks)
            collected += action_chunks.shape[0]
            if collected >= args.max_embed_samples:
                break

        if len(all_chunks) == 0:
            print("  No sampled chunks, skipping embedding visualization")
        else:
            all_chunks = torch.cat(all_chunks, dim=0)[:args.max_embed_samples]
            all_chunks = all_chunks.to(device=device, dtype=torch.float32)

            print(f"  Collected {all_chunks.shape[0]} action chunks")
            embeddings = collect_embeddings(model, all_chunks, max_samples=args.max_embed_samples)

            label = cfg.get('exp_name', 'all')
            if args.mode in ('embedding', 'both'):
                fname = plot_embedding_distribution(
                    embeddings, label, output_dir,
                    perplexity=args.tsne_perplexity,
                )
                print(f"    Saved: {fname}")

            if args.mode in ('embedding_umap', 'both_umap'):
                fname = plot_embedding_distribution_umap(
                    embeddings, label, output_dir,
                )
                print(f"    Saved: {fname}")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize diffusion token usage and embedding distributions")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--steps', type=int, default=20, help='Diffusion sampling steps')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_episodes_per_dataset', type=int, default=100,
                        help='Max episodes to visualize per dataset')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['tokens', 'embedding', 'embedding_umap', 'both', 'both_umap'],
                        help='Which visualization to run')
    parser.add_argument('--max_embed_samples', type=int, default=4096,
                        help='Max action chunks to sample for embedding visualization')
    parser.add_argument('--tsne_perplexity', type=float, default=30,
                        help='t-SNE perplexity')
    args = parser.parse_args()

    main(args)
