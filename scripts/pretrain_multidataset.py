import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import tqdm
import hydra
import math
import wandb
import os
import sys
import random
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)

from catok.models.catok_ddt.vanilla_utils import construct_model, construct_dataloader
from catok.models.catok_ddt.vanilla_tokenizer import VanillaVQDiffusion
from catok.training.rlds_dataset import DistributedWeightedSampler


@hydra.main(config_path="../conf/config_libero_bridge", config_name="vanilla_diffusion", version_base="1.3")
def pretrain(cfg: DictConfig):
    
    # Set seed for reproducibility
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Check if distributed or single GPU mode
    use_ddp = cfg.get('use_ddp', True)

    if use_ddp:
        # +++ DDP Imports +++
        import torch.distributed as dist
        from accelerate import PartialState
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler
        
        # Initialize distributed environment using accelerate.PartialState
        distributed_state = PartialState()
        device_id = distributed_state.local_process_index
        torch.cuda.set_device(device_id)
        device = f'cuda:{device_id}'
        is_main_process = distributed_state.is_main_process
        world_size = distributed_state.num_processes
        rank = distributed_state.process_index
    else:
        # Single GPU mode
        device_id = 0
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        is_main_process = True
        world_size = 1
        rank = 0
        distributed_state = None

    cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    
    # model
    vq_mode = cfg['tokenizer']['vq']['vq_mode']
    max_reset_steps = cfg['tokenizer']['vq']['max_reset_steps']
    model = construct_model(cfg['tokenizer'], device=device)
    # Load pretrained weights if init_from_ckpt is specified
    init_from = cfg.get('init_from_ckpt', None)
    if init_from:
        ckpt = torch.load(init_from, map_location=device)
        # Rebuild model using checkpoint's own config to guarantee architecture match,
        # then copy weights. This avoids d_vq/add_vq_latent mismatches.
        base_tok_cfg = dict(ckpt['config']['tokenizer'])
        # Preserve infonce/or_direct settings from the current run config
        for key in ('adjacent_infonce', 'or_direct', 'overlap_infonce'):
            if key in cfg['tokenizer']:
                base_tok_cfg[key] = cfg['tokenizer'][key]
        model = construct_model(base_tok_cfg, device=device)
        result = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f'[init_from_ckpt] loaded {init_from}, missing={result.missing_keys[:5]}, unexpected={result.unexpected_keys[:5]}')
    model.train()
    if use_ddp:
        model = DDP(
            model, 
            device_ids=[device_id], 
            find_unused_parameters=True, 
            gradient_as_bucket_view=True
        )
    save_dir = cfg['ckpt_dir']
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    # dataloader
    per_gpu_batch_size = cfg['train']['batch_size'] // world_size
    assert cfg['train']['batch_size'] % world_size == 0, \
        f"batch_size ({cfg['train']['batch_size']}) must be divisible by world_size ({world_size})"

    # Use sequential (non-shuffled) loader when adjacent_infonce or or_direct is active so that
    # batch[i] and batch[i+1] are truly consecutive windows from the same episode.
    adjacent_infonce_weight = float(
        cfg.get('tokenizer', {}).get('adjacent_infonce', {}).get('weight', 0.0) or 0.0
    )
    or_direct_weight = float(
        cfg.get('tokenizer', {}).get('or_direct', {}).get('weight', 0.0) or 0.0
    )
    use_sequential_for_adjacent = (adjacent_infonce_weight > 0.0) or (or_direct_weight > 0.0)
    train_sampler, train_loader, _ = construct_dataloader(
        cfg,
        batch_size=per_gpu_batch_size,
        distributed=use_ddp,
        rank=rank if use_ddp else None,
        world_size=world_size if use_ddp else None,
        drop_last=use_ddp,
        eval_mode=use_sequential_for_adjacent,
    )
    
    # loss function
    if cfg['train']['loss_fn'] == 'l1':
        loss_fn = nn.L1Loss()
    elif cfg['train']['loss_fn'] == 'l2':
        loss_fn = nn.MSELoss()
    else:
        raise ValueError('Unknown loss function.')
    
    # optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['learning_rate'],
        weight_decay=cfg['train']['weight_decay']
    )
    num_training_steps = cfg['train']['epochs'] * len(train_loader)
    warmup = cfg['train'].get('warmup', 0)
    warmup_steps = int(num_training_steps * warmup / 100.0) if warmup > 0 else 0
    def combine_scheduler(current_step: int):
        if current_step < warmup_steps:
            # linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine Annealing after warmup
        cosine_steps = num_training_steps - warmup_steps
        step_in_cosine = current_step - warmup_steps

        return 0.5 * (1 + math.cos(math.pi * step_in_cosine / cosine_steps))
    scheduler = LambdaLR(
        optimizer, 
        lr_lambda=combine_scheduler
    )

    # logger
    exp_name = cfg["wandb_expname"]
    if is_main_process and cfg['use_wandb']:
        wandb.init(name=exp_name, project=cfg['wandb_project'], config=cfg)

    epochs = cfg['train']['epochs']
    dct_loss = cfg['train']['dct_loss']
    # Adjacent InfoNCE weight scheduling: optionally cosine-decay from initial weight to 0
    adj_cfg = cfg['tokenizer'].get('adjacent_infonce', {}) or {}
    infonce_decay_steps = int(adj_cfg.get('decay_steps', 0))  # 0 = no decay
    infonce_init_weight = float(adj_cfg.get('weight', 0.0))
    consistency_weight = cfg['tokenizer'].get('decoder', {}).get('vanilla_consistency_weight', 0.0)
    consistency_enabled = consistency_weight > 0
    
    if is_main_process:
        print("DataLoader Len: ", len(train_loader))
    steps = 0
    for epoch in tqdm.trange(epochs, disable=(not is_main_process)):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if isinstance(train_sampler, DistributedWeightedSampler):
            train_sampler.set_epoch(epoch)
                
        for batch in tqdm.tqdm(train_loader, disable=(not is_main_process)):
            # batch_x shape: (B, horizon, action_dim)
            steps += 1
            optimizer.zero_grad()
            # Cosine-decay infonce weight if decay_steps is set
            if infonce_decay_steps > 0 and infonce_init_weight > 0:
                t = min(steps / infonce_decay_steps, 1.0)
                cur_weight = infonce_init_weight * 0.5 * (1 + math.cos(math.pi * t))
                m = model.module if hasattr(model, 'module') else model
                m.adjacent_infonce_weight = cur_weight
            batch = batch.to(device=device, dtype=torch.float32)
            # Build adjacent pairs from batch for adjacent InfoNCE (x[i] vs x[i+1]).
            x_next = batch[1:] if batch.shape[0] > 1 else None
            x_curr = batch[:-1] if x_next is not None else batch
            recon_x, loss, loss_dict, indices, perplexity = model(x_curr, x_next=x_next)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if vq_mode in ['RVQ_EMA', 'VQ_EMA'] and steps == max_reset_steps:
                if use_ddp:
                    model.module.vq.set_threshold(0)
                else:
                    model.vq.set_threshold(0)

            if is_main_process and cfg['use_wandb'] and steps % cfg.get('log_every_n_steps', 10) == 0:
                log_dict = {"recon_loss": loss_dict['recon']}
                if vq_mode == 'RVQ_EMA':
                    log_dict["vq_loss"] = loss_dict['vq_loss']
                    for i in range(len(indices)):
                        log_dict[f"active_code_{i}"] = len(torch.unique(indices[i]))
                        log_dict[f"perplexity_{i}"] = perplexity[i]
                elif vq_mode is not None:
                    log_dict["vq_loss"] = loss_dict['vq_loss']
                    log_dict["active_code"] = len(torch.unique(indices))
                    log_dict["perplexity"] = perplexity
                if dct_loss:
                    log_dict["dct_loss"] = loss_dict["dct"]
                if consistency_enabled and "consistency" in loss_dict:
                    log_dict["consistency_loss"] = loss_dict["consistency"]
                if "overlap_infonce" in loss_dict:
                    log_dict["overlap_infonce"] = loss_dict["overlap_infonce"]
                if "adjacent_infonce" in loss_dict:
                    log_dict["adjacent_infonce"] = loss_dict["adjacent_infonce"]
                if "or_direct" in loss_dict:
                    log_dict["or_direct"] = loss_dict["or_direct"]
                wandb.log(log_dict, step=steps)

            if is_main_process and steps % cfg['save_every_n_steps'] == 0:
                # Get model state dict (handle DDP wrapper)
                model_state = model.module.state_dict() if use_ddp else model.state_dict()
                save_path = os.path.join(save_dir, f"{steps:06d}.pth")
                torch.save({
                    'model_state_dict': model_state,
                    'config': cfg,
                },
                    save_path,
                )

    # Finish wandb logging before cleanup
    if is_main_process and cfg.get('use_wandb', False):
        wandb.finish()
        
    # Clean up distributed process group (always called, even on exception)
    if use_ddp:
        if dist.is_initialized():
            print("Destroying process group")
            dist.destroy_process_group()


if __name__ == "__main__":
    pretrain()