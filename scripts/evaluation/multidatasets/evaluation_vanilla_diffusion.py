import random
import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple

from catok.models.catok_ddt.vanilla_utils import load_checkpoint
from catok.training.statistics import build_action_normalizer
from catok.training.rlds_dataset import make_rlds_dataloader

from tqdm import tqdm
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import os

import tensorflow_datasets as tfds
import tensorflow as tf

@torch.no_grad()
def calculate_codebook_entropy(indices, num_embeddings):
    flat_indices = indices.reshape(-1)
    counts = torch.bincount(flat_indices, minlength=num_embeddings).float()
    probs = counts / torch.sum(counts)
    probs = probs + 1e-10
    entropy_abs = -torch.sum(probs * torch.log(probs))

    # entropy normalization
    max_entropy = torch.log(torch.tensor(num_embeddings, dtype=torch.float))
    entropy_norm = entropy_abs / max_entropy

    perplexity = torch.exp(entropy_abs)
    return entropy_norm.item(), perplexity.item(), counts


def _unwrap_rlds_state_action_dataset(dataset) -> Optional[Any]:
    """Innermost RLDSStateActionDataset from ActionTransformDataset wrapping."""
    from catok.training.rlds_dataset import RLDSStateActionDataset

    d = dataset
    while hasattr(d, "dataset"):
        d = d.dataset
    if isinstance(d, RLDSStateActionDataset):
        return d
    return None


@torch.no_grad()
def _encode_vq_indices(model, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Discretize actions to (B, num_quantizers, seq_len). Uses model.encoding if available."""
    if getattr(model, "vq_mode", None) is None:
        return None
    if hasattr(model, "encoding"):
        return model.encoding(x)
    z = model.encoder(x)
    _, _, indices, _ = model.vq(z)
    if isinstance(indices, list):
        indices = torch.stack(indices, dim=1)
    return indices


def _collect_in_episode_adjacent_pairs(
    rlds,
    max_pairs: Optional[int],
    max_episodes: Optional[int],
    seed: int,
) -> List[Tuple[int, int]]:
    """Indices (i, j) into the same dataset for windows (ep, t) and (ep, t+1)."""
    pairs: List[Tuple[int, int]] = []
    for k in range(len(rlds.samples) - 1):
        a, b = rlds.samples[k], rlds.samples[k + 1]
        if a[0] != b[0] or a[1] + 1 != b[1]:
            continue
        if max_episodes is not None and a[0] >= max_episodes:
            continue
        pairs.append((k, k + 1))
    rng = random.Random(seed)
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = rng.sample(pairs, max_pairs)
    return pairs


def _default_stats_path_map() -> Dict[str, str]:
    return {
        "bridge": "data/bridge/stats.json",
        "Libero_RLDS/libero_spatial_no_noops": "data/Libero_RLDS/libero_spatial_no_noops/stats.json",
        "Libero_RLDS/libero_goal_no_noops": "data/Libero_RLDS/libero_goal_no_noops/stats.json",
        "Libero_RLDS/libero_object_no_noops": "data/Libero_RLDS/libero_object_no_noops/stats.json",
        "Libero_RLDS/libero_10_no_noops": "data/Libero_RLDS/libero_10_no_noops/stats.json",
    }


def _libero_dataset_specs(weights: Tuple[float, float, float, float] = (5.0, 5.0, 5.0, 5.0)) -> List[Tuple[str, float]]:
    k = ["Libero_RLDS/libero_spatial_no_noops", "Libero_RLDS/libero_goal_no_noops",
         "Libero_RLDS/libero_object_no_noops", "Libero_RLDS/libero_10_no_noops"]
    return list(zip(k, weights))


def _eval_loader_bundle(
    eval_subset: str,
) -> Tuple[List[Tuple[str, float]], Dict[str, str], Dict[str, str]]:
    stats_map_full = _default_stats_path_map()
    if eval_subset == "all":
        dataset_specs = [
            ("bridge", 1.0),
            ("Libero_RLDS/libero_spatial_no_noops", 5.0),
            ("Libero_RLDS/libero_goal_no_noops", 5.0),
            ("Libero_RLDS/libero_object_no_noops", 5.0),
            ("Libero_RLDS/libero_10_no_noops", 5.0),
        ]
        libero_names = [
            "Libero_RLDS/libero_spatial_no_noops",
            "Libero_RLDS/libero_goal_no_noops",
            "Libero_RLDS/libero_object_no_noops",
            "Libero_RLDS/libero_10_no_noops",
        ]
        normalizer_method_map = {"bridge": "bridge", **{n: "qq" for n in libero_names}}
        spm = stats_map_full
    elif eval_subset == "libero":
        dataset_specs = _libero_dataset_specs()
        spm = {k: stats_map_full[k] for k, _ in dataset_specs}
        libero_only = list(spm.keys())
        normalizer_method_map = {n: "qq" for n in libero_only}
    else:
        raise ValueError(f"Unknown eval_subset {eval_subset!r}; use 'all' or 'libero'.")

    assert all(k in stats_map_full for k, _ in dataset_specs), "stats map missing subset key"
    return dataset_specs, spm, normalizer_method_map



@torch.no_grad()
def run_overlap_rate_evaluation(
    model,
    device: str,
    data_root: str,
    action_horizon: int,
    target_action_dim: int,
    dataset_name: str = "bridge",
    max_pairs: int = 2048,
    max_episodes: Optional[int] = 32,
    pair_batch: int = 64,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sample in-episode consecutive action windows, encode to VQ indices, and
    report overlap rate (OR) as in ActionCodec: agreement on the overlap region
    (temporal mode) or per-slot (Dual encoder).
    """
    stats_map = _default_stats_path_map()
    if dataset_name not in stats_map:
        raise ValueError(
            f"Unknown dataset_name {dataset_name!r}; choose one of {list(stats_map)}"
        )

    loader_cfg = {
        "dataset_specs": [(dataset_name, 1.0)],
        "data_root": data_root,
        "horizon": action_horizon,
        "batch_size": 1,
        "action_only": True,
        "num_parallel_reads": 4,
        "num_workers": 0,
        "target_action_dim": target_action_dim,
        "stats_path_map": {dataset_name: stats_map[dataset_name]},
        "normalizer_method": "qq",
        "normalizer_config": {
            "clip": True,
            "action_dim": 7,
        },
        "eval_mode": True,
    }
    if max_episodes is not None:
        loader_cfg["max_episodes"] = max_episodes

    _, _, concat = make_rlds_dataloader(**loader_cfg)
    if len(concat.datasets) < 1:
        return {"error": "empty dataset"}

    wrapped = concat.datasets[0]
    rlds = _unwrap_rlds_state_action_dataset(wrapped)
    if rlds is None:
        return {"error": "could not unwrap RLDSStateActionDataset"}

    pair_indices = _collect_in_episode_adjacent_pairs(
        rlds, max_pairs, max_episodes, seed
    )
    if not pair_indices:
        return {
            "error": "no adjacent window pairs (check data path / max_episodes)",
            "num_pairs": 0,
        }

    tot_match = 0.0
    tot_positions = 0
    per_q_sum: Optional[torch.Tensor] = None
    tot_m_elements = 0
    report_mode: Optional[str] = None

    for s in range(0, len(pair_indices), pair_batch):
        chunk = pair_indices[s : s + pair_batch]
        i0 = [a for a, _ in chunk]
        i1 = [b for _, b in chunk]
        t0 = torch.stack(
            [torch.as_tensor(wrapped[i], dtype=torch.float32) for i in i0]
        ).to(device)
        t1 = torch.stack(
            [torch.as_tensor(wrapped[i], dtype=torch.float32) for i in i1]
        ).to(device)

        c0 = _encode_vq_indices(model, t0)
        c1 = _encode_vq_indices(model, t1)
        if c0 is None or c1 is None:
            return {
                "error": "model has no VQ / could not produce indices",
                "num_pairs": 0,
            }
        # Match main eval: single VQ layer is (B, S); RVQ is (B, Q, S).
        if c0.dim() == 2:
            c0 = c0.unsqueeze(1)
            c1 = c1.unsqueeze(1)
        h_batch = t0.shape[1]
        _, Q, S = c0.shape
        if S == h_batch:
            m = c0[:, :, 1:] == c1[:, :, :-1]
            if report_mode is None:
                report_mode = "temporal"
        else:
            m = c0 == c1
            if report_mode is None:
                report_mode = "slot"
        joint = m.all(dim=1)
        tot_match += joint.float().sum().item()
        tot_positions += joint.numel()
        tot_m_elements += m.shape[0] * m.shape[2]
        sum_q = m.float().sum(dim=(0, 2))
        if per_q_sum is None:
            per_q_sum = torch.zeros(Q, device=sum_q.device, dtype=sum_q.dtype)
        per_q_sum += sum_q

    assert per_q_sum is not None and report_mode is not None
    per_q_mean = (per_q_sum / max(1, tot_m_elements)).cpu().numpy().tolist()
    out: Dict[str, Any] = {
        "mode": report_mode,
        "overlap_rate_joint": tot_match / max(1, tot_positions),
        "overlap_rate_per_quantizer": per_q_mean,
        "num_adjacent_window_pairs": len(pair_indices),
        "pair_batch": pair_batch,
        "action_horizon": action_horizon,
        "token_seq_len": int(c0.shape[2]) if c0 is not None else None,
        "num_quantizers": int(c0.shape[1]) if c0 is not None else None,
        "dataset": dataset_name,
        "max_episodes_capped": max_episodes,
        "description": (
            "temporal: compare c[...,1:] vs next[...,:-1] when seq_len==horizon; "
            "slot: full sequence match when seq_len!=horizon (e.g. Dual encoder)"
        ),
    }
    return out


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluation(args):
    set_seed(42)
    model_path = args.model_path
    device = args.device

    model, cfg = load_checkpoint(model_path, device)
    model.eval()
    
    data_cfg = cfg['data']
    action_dim = cfg['tokenizer']['basic']['action_dim']
    dataset_specs, stats_path_map_sp, normalizer_method_map = _eval_loader_bundle(
        getattr(args, "eval_subset", "all")
    )
    loader_cfg = {
        'dataset_specs': dataset_specs,
        'data_root': data_cfg['root_dir'],
        'horizon': data_cfg['horizon'],
        'batch_size': 4096,
        'action_only': True,
        'num_parallel_reads': 8,
        'num_workers': 16,
        'target_action_dim': action_dim,
        'stats_path_map': stats_path_map_sp,
        'normalizer_method': 'qq',
        'normalizer_method_map': normalizer_method_map,
        'normalizer_config': {
            'clip': True,
            'action_dim': action_dim,
        },
    }
    sampler, loader, _ = make_rlds_dataloader(
        **loader_cfg
    )

    # reconstruction loss
    current_l1_mean = 0.
    current_l2_mean = 0.
    current_n = 0

    # inference time
    encode_inference_times = []
    total_encode_time = 0.  
    decode_inference_times = []
    total_decode_time = 0.
    num_batches = 0
    
    # codebook entropy
    all_vq_indices = []

    pbar = tqdm(loader)
    for action_chunks in pbar:
        with torch.no_grad():
            action_chunks = action_chunks.to(device=device, dtype=torch.float32)

            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            if device == 'cuda':
                torch.cuda.synchronize()
            encode_start_time = time.perf_counter()

            z = model.encoder(action_chunks)

            if device == 'cuda':
                torch.cuda.synchronize()
            encode_end_time = time.perf_counter()

            if model.vq_mode is not None:
                vq_loss, z_q, indices, perplexity = model.vq(z)
                
                if isinstance(indices, list):
                    stacked_indices = torch.stack(indices, dim=1) # (Batch, Num_Quantizers, Seq)
                else:
                    stacked_indices = indices.unsqueeze(1)        # (Batch, 1, Seq)
                
                all_vq_indices.append(stacked_indices.cpu())
            else:
                z_q = z

            if device == 'cuda':
                torch.cuda.synchronize()
            decoder_start_time = time.perf_counter()

            if args.sampling_method == "decoder_one_step":
                pred_actions, _ = model.decoder.forward_one_step_decoder(encoder_hidden_states=z_q, context_see_xt=model.context_see_xt)
            else:
                pred_actions = model.flow.sample(
                    action_chunks.shape,
                    z_q,
                    steps=args.steps,
                    one_step=args.flow_one_step,
                    solver=args.flow_solver,
                    full_tokens=args.flow_full_tokens,
                )


            if device == 'cuda':
                torch.cuda.synchronize()
            decode_end_time = time.perf_counter()

            l1 = F.l1_loss(pred_actions, action_chunks)
            l2 = F.mse_loss(pred_actions, action_chunks)

            batch_encode_time = encode_end_time - encode_start_time
            batch_decode_time = decode_end_time - decoder_start_time
            encode_inference_times.append(batch_encode_time)
            decode_inference_times.append(batch_decode_time)
            total_encode_time += batch_encode_time
            total_decode_time += batch_decode_time
            num_batches += 1

        new_n = len(action_chunks)
        current_n += new_n
        current_l1_mean = current_l1_mean + (l1 - current_l1_mean) * new_n / current_n
        current_l2_mean = current_l2_mean + (l2 - current_l2_mean) * new_n / current_n
        pbar.set_postfix(batch_l1=f"{l1.item():.6f}")

    # 计算推理时间统计
    encode_inference_times = np.array(encode_inference_times)
    decode_inference_times = np.array(decode_inference_times)
    avg_encode_time = np.mean(encode_inference_times)
    std_encode_time = np.std(encode_inference_times)
    min_encode_time = np.min(encode_inference_times)
    max_encode_time = np.max(encode_inference_times)

    avg_decode_time = np.mean(decode_inference_times)
    std_decode_time = np.std(decode_inference_times)
    min_decode_time = np.min(decode_inference_times)
    max_decode_time = np.max(decode_inference_times)

    print(f'L1 loss: {current_l1_mean:.6f}')
    print(f'L2 loss: {current_l2_mean:.6f}')
    print(f'Average encode time per batch: {avg_encode_time*1000:.2f} ms')
    print(f'Average decode time per batch: {avg_decode_time*1000:.2f} ms')
    print(f'Total encode time: {total_encode_time:.2f} s')
    print(f'Total decode time: {total_decode_time:.2f} s')
    
    # Compute codebook usage statistics
    codebook_stats = {}
    if model.vq_mode is not None and len(all_vq_indices) > 0:
        full_indices = torch.cat(all_vq_indices, dim=0) # (Total_Samples, Num_Quantizers, Seq)
        num_quantizers = full_indices.shape[1]
        num_embeddings = model.vq.num_embeddings

        print("\n" + "="*40)
        print("Codebook Usage Report (Full Test Set)")
        print("="*40)
        
        for i in range(num_quantizers):
            layer_indices = full_indices[:, i, :]
            ent, perp, counts = calculate_codebook_entropy(layer_indices, num_embeddings)
            unused_codes = (counts == 0).sum().item()
            
            print(f"Quantizer Layer {i}:")
            print(f"  - Entropy:      {ent:.4f}")
            print(f"  - Perplexity:   {perp:.2f} / {num_embeddings}")
            print(f"  - Unused Codes: {unused_codes} ({unused_codes/num_embeddings*100:.1f}%)")
            
            codebook_stats[f'layer_{i}'] = {
                'entropy': ent,
                'perplexity': perp,
                'unused_codes': unused_codes,
                'total_codes': num_embeddings
            }
        print("="*40 + "\n")

    overlap_rate_stats: Dict[str, Any] = {}
    if getattr(args, "compute_overlap_rate", False):
        overlap_rate_stats = run_overlap_rate_evaluation(
            model,
            device,
            data_cfg["root_dir"],
            action_horizon=data_cfg["horizon"],
            target_action_dim=action_dim,
            dataset_name=getattr(args, "overlap_dataset", "bridge"),
            max_pairs=getattr(args, "overlap_max_pairs", 2048),
            max_episodes=getattr(args, "overlap_max_episodes", 32),
            pair_batch=getattr(args, "overlap_pair_batch", 64),
            seed=42,
        )
        print("\n" + "=" * 40)
        print("Overlap rate (in-episode adjacent sliding windows)")
        print("=" * 40)
        if "error" in overlap_rate_stats:
            print(overlap_rate_stats["error"])
        else:
            print(
                f"  mode: {overlap_rate_stats.get('mode')}, "
                f"OR (joint): {overlap_rate_stats.get('overlap_rate_joint', 0):.4f}, "
                f"pairs: {overlap_rate_stats.get('num_adjacent_window_pairs', 0)}"
            )
            if overlap_rate_stats.get("overlap_rate_per_quantizer"):
                pq = [
                    round(x, 4)
                    for x in overlap_rate_stats["overlap_rate_per_quantizer"]
                ]
                print(f"  per quantizer: {pq}")
        print("=" * 40 + "\n")

    loss = {
        'eval_subset': getattr(args, 'eval_subset', 'all'),
        'checkpoint': str(Path(model_path).resolve()),
        'l1 loss': current_l1_mean.item(),
        'l2 loss': current_l2_mean.item(),
        'inference_time': {
            'avg_per_batch_encode_ms': float(avg_encode_time * 1000),
            'std_per_batch_encode_ms': float(std_encode_time * 1000),
            'min_per_batch_encode_ms': float(min_encode_time * 1000),
            'max_per_batch_encode_ms': float(max_encode_time * 1000),
            'total_encode_seconds': float(total_encode_time),
            'avg_per_batch_decode_ms': float(avg_decode_time * 1000),
            'std_per_batch_decode_ms': float(std_decode_time * 1000),
            'min_per_batch_decode_ms': float(min_decode_time * 1000),
            'max_per_batch_decode_ms': float(max_decode_time * 1000),
            'total_decode_seconds': float(total_decode_time),
            'num_batches': int(num_batches),
            'total_samples': int(current_n),
        },
        # ================= 新增：保存到 JSON =================
        'codebook_stats': codebook_stats
        # =====================================================
    }
    if overlap_rate_stats:
        loss["overlap_rate"] = overlap_rate_stats

    ckpt_dir = Path(model_path).parent
    subset_tag = '_libero' if getattr(args, 'eval_subset', 'all') == 'libero' else ''
    if args.sampling_method == "decoder_one_step":
        fname = f'{ckpt_dir}/eval_decoder_one_step{subset_tag}.json'
    else:
        one_step_tag = "_one_step" if args.flow_one_step else ""
        full_tokens_tag = "_fulltokens" if args.flow_full_tokens else ""
        fname = f'{ckpt_dir}/eval_flow_{args.flow_solver}_{args.steps}steps{one_step_tag}{full_tokens_tag}{subset_tag}.json'
    with open(fname, 'w') as f:
        json.dump(loss, f, indent=4)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Causal Action Tokenizer Reconstruction")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--steps', type=int, default=1, help='Diffusion steps')
    parser.add_argument(
        '--sampling_method',
        type=str,
        default='flow',
        choices=['flow', 'decoder_one_step'],
        help='Sampling method: flow sampler or decoder one-step reconstruction',
    )
    parser.add_argument(
        '--flow_solver',
        type=str,
        default='euler',
        choices=['euler', 'heun'],
        help='Flow ODE solver (only used when sampling_method=flow)',
    )
    parser.add_argument(
        '--flow_one_step',
        action='store_true',
        help='Force flow sampler to run only one ODE step (only used when sampling_method=flow)',
    )
    parser.add_argument(
        '--flow_full_tokens',
        action='store_true',
        help='Use full-token conditioning mask during flow inference (only used when sampling_method=flow)',
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the evaluation on')
    parser.add_argument(
        '--eval-subset',
        type=str,
        default='all',
        choices=('all', 'libero'),
        help='Datasets for reconstruction: full training mix (bridge+Libero) or Libero RLDS only.',
    )
    parser.add_argument(
        '--one_step',
        action='store_true',
        help='[Deprecated] Same as --sampling_method decoder_one_step',
    )
    parser.add_argument(
        '--compute-overlap-rate',
        action='store_true',
        help='After reconstruction eval, sample in-episode adjacent action windows and '
        'report ActionCodec-style overlap rate on VQ indices (see run_overlap_rate_evaluation).',
    )
    parser.add_argument(
        '--overlap-only',
        action='store_true',
        help='Only run overlap-rate on sampled chunks (skip full L1/L2 pass). Writes eval_overlap_rate.json.',
    )
    parser.add_argument(
        '--overlap-dataset',
        type=str,
        default='bridge',
        help='RLDS dataset key for OR (must be in stats map, e.g. bridge, Libero_RLDS/...).',
    )
    parser.add_argument(
        '--overlap-max-pairs',
        type=int,
        default=2048,
        help='Max (A_t, A_{t+1}) pairs to average; subsampled at random if more exist.',
    )
    parser.add_argument(
        '--overlap-max-episodes',
        type=int,
        default=32,
        help='Cap episodes loaded for OR (-1 = no cap on episode index / load all in RLDS).',
    )
    parser.add_argument(
        '--overlap-pair-batch',
        type=int,
        default=64,
        help='Batch size in number of adjacent pairs per encode step.',
    )
    args = parser.parse_args()
    if args.one_step:
        args.sampling_method = 'decoder_one_step'

    def _resolve_overlap_max_episodes(v: int):
        return None if v < 0 else v

    if args.overlap_only:
        set_seed(42)
        model_path = args.model_path
        device = args.device
        model, cfg = load_checkpoint(model_path, device)
        model.eval()
        data_cfg = cfg['data']
        action_dim = cfg['tokenizer']['basic']['action_dim']
        overlap_stats = run_overlap_rate_evaluation(
            model,
            device,
            data_cfg['root_dir'],
            action_horizon=data_cfg['horizon'],
            target_action_dim=action_dim,
            dataset_name=args.overlap_dataset,
            max_pairs=args.overlap_max_pairs,
            max_episodes=_resolve_overlap_max_episodes(args.overlap_max_episodes),
            pair_batch=args.overlap_pair_batch,
            seed=42,
        )
        ckpt_dir = Path(model_path).parent
        out_path = ckpt_dir / 'eval_overlap_rate.json'
        with open(out_path, 'w') as f:
            json.dump(overlap_stats, f, indent=4)
        print(f"Wrote {out_path}")
        if "error" not in overlap_stats:
            print(
                f"OR (joint): {overlap_stats.get('overlap_rate_joint', 0):.4f} "
                f"({overlap_stats.get('mode')}, {overlap_stats.get('num_adjacent_window_pairs', 0)} pairs)"
            )
    else:
        args.overlap_max_episodes = _resolve_overlap_max_episodes(args.overlap_max_episodes)
        evaluation(args)
