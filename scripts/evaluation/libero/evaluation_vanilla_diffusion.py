import random
import torch
import torch.nn.functional as F
import numpy as np
import time

from catok.models.catok_ddt.vanilla_utils import load_checkpoint
from catok.training.statistics import build_action_normalizer
from catok.training.dataset import make_dataset

from tqdm import tqdm
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import os

import tensorflow_datasets as tfds
import tensorflow as tf


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
    
    action_dim = cfg['tokenizer']['basic']['action_dim']
    def fetch_func(x):
        x = x[:, :, :-1]
        # padding to action_dim
        x = np.pad(x, ((0, 0), (0, 0), (0, action_dim - x.shape[-1])), mode='constant')
        return x
    normalizer_cfg = cfg['normalizer']
    normalizer = build_action_normalizer(
        method=normalizer_cfg['type'],
        normalizer_config=normalizer_cfg['config']
    )
    dataset = make_dataset(cfg['dataset']['dataset_name'], cfg['dataset']['dataset_config'])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4096,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )

    # reconstruction loss
    current_l1_mean = 0.
    current_l2_mean = 0.
    current_n = 0

    # inference time
    inference_times = []
    total_inference_time = 0.
    num_batches = 0

    for batch in tqdm(loader):
        with torch.no_grad():
            action_chunks = fetch_func(batch)
            action_chunks = normalizer.normalize(action_chunks)
            action_chunks = torch.from_numpy(action_chunks)
            action_chunks = action_chunks.to(device=device, dtype=torch.float32)

            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # reconstruction absolute action chunk
            pred_actions = model.reconstruct(action_chunks, steps=args.steps, one_step=args.one_step)

            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            l1 = F.l1_loss(pred_actions, action_chunks)
            l2 = F.mse_loss(pred_actions, action_chunks)

            batch_inference_time = end_time - start_time
            inference_times.append(batch_inference_time)
            total_inference_time += batch_inference_time
            num_batches += 1

        new_n = len(action_chunks)
        current_n += new_n
        current_l1_mean = current_l1_mean + (l1 - current_l1_mean) * new_n / current_n
        current_l2_mean = current_l2_mean + (l2 - current_l2_mean) * new_n / current_n

    # 计算推理时间统计
    inference_times = np.array(inference_times)
    avg_inference_time = np.mean(inference_times)  # 平均每个batch的推理时间
    std_inference_time = np.std(inference_times)   # 推理时间的标准差
    min_inference_time = np.min(inference_times)   # 最小推理时间
    max_inference_time = np.max(inference_times)   # 最大推理时间
    throughput = current_n / total_inference_time   # 每秒处理的样本数

    print(f'L1 loss: {current_l1_mean:.6f}')
    print(f'L2 loss: {current_l2_mean:.6f}')
    print(f'Average inference time per batch: {avg_inference_time*1000:.2f} ms')
    print(f'Inference time std: {std_inference_time*1000:.2f} ms')
    print(f'Min inference time: {min_inference_time*1000:.2f} ms')
    print(f'Max inference time: {max_inference_time*1000:.2f} ms')
    print(f'Total inference time: {total_inference_time:.2f} s')
    print(f'Throughput: {throughput:.2f} samples/second')
    print(f'Batch size: {args.batch_size if hasattr(args, "batch_size") else 4096}')
    
    loss = {
        'l1 loss': current_l1_mean.item(),
        'l2 loss': current_l2_mean.item(),
        'inference_time': {
            'avg_per_batch_ms': float(avg_inference_time * 1000),
            'std_per_batch_ms': float(std_inference_time * 1000),
            'min_per_batch_ms': float(min_inference_time * 1000),
            'max_per_batch_ms': float(max_inference_time * 1000),
            'total_seconds': float(total_inference_time),
            'throughput_samples_per_second': float(throughput),
            'num_batches': int(num_batches),
            'total_samples': int(current_n),
        }
    }

    ckpt_dir = Path(model_path).parent
    if args.one_step:
        fname = f'{ckpt_dir}/eval_one_step.json'
    else:
        fname = f'{ckpt_dir}/eval_{args.steps}steps.json'
    with open(fname, 'w') as f:
        json.dump(loss, f, indent=4)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Causal Action Tokenizer Reconstruction")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--steps', type=int, default=20, help='Diffusion steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the evaluation on')
    parser.add_argument('--one_step', action='store_true', help='Whether to use one-step reconstruction')
    args = parser.parse_args()
    
    evaluation(args)