import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import tqdm
import math
import wandb
import os
import sys
import random
from omegaconf import DictConfig, OmegaConf
import argparse

from catok.models.catok_ddt.vanilla_utils import load_checkpoint, construct_dataloader
from catok.training.rlds_dataset import DistributedWeightedSampler


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune Decoder while freezing Encoder and VQ")
    
    # Required arguments
    parser.add_argument("--ckpt", type=str, required=True, 
                        help="Path to the checkpoint file (.pth)")
    
    # Training hyperparameters
    parser.add_argument("--warmup_steps", type=int, default=1000, 
                        help="Number of steps for learning rate warmup")
    parser.add_argument("--epochs", type=int, default=800, 
                        help="Total number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate for the Decoder")
    parser.add_argument("--weight_decay", type=float, default=1e-2, 
                        help="Weight decay for the optimizer")
    parser.add_argument("--batch_size", type=int, default=512, 
                        help="Batch size for training")
    parser.add_argument("--use_ddp", action='store_true', 
                        help="Whether to use distributed data parallel")

    args = parser.parse_args()
    return args

def finetune(args):
    
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Check if distributed or single GPU mode
    use_ddp = args.use_ddp

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
    
    # model
    model, cfg = load_checkpoint(args.ckpt, device=device)
    vq_mode = cfg['tokenizer']['vq']['vq_mode']
    model.encoder.eval()
    model.vq.eval()
    model.decoder.train()
    # model.enc_input.requires_grad_(False)
    # model.pos_embedding.requires_grad_(False)
    # model.query_tokens.requires_grad_(False)
    model.encoder.requires_grad_(False)
    model.vq.requires_grad_(False)
    model.decoder.requires_grad_(True)
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
    train_sampler, train_loader, _ = construct_dataloader(
        cfg,
        batch_size=cfg['train']['batch_size'],
        distributed=use_ddp,
        rank=rank if use_ddp else None,
        world_size=world_size if use_ddp else None,
        drop_last=use_ddp,
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
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * world_size,
        weight_decay=args.weight_decay
    )
    num_training_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_steps
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
    exp_name = cfg["wandb_expname"] + '_onestep'
    if is_main_process and cfg['use_wandb']:
        wandb.init(name=exp_name, project=cfg['wandb_project'], config=cfg)

    epochs = args.epochs
    dct_loss = cfg['train']['dct_loss']
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
            
            batch = batch.to(device=device, dtype=torch.float32)           
            recon_x, loss, loss_dict, indices, perplexity = model(batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

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
                wandb.log(log_dict, step=steps)

            if is_main_process and steps % cfg['save_every_n_steps'] == 0:
                # Get model state dict (handle DDP wrapper)
                model_state = model.module.state_dict() if use_ddp else model.state_dict()
                save_path = os.path.join(save_dir, f"onestep_{steps:06d}.pth")
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
    finetune(args=get_args())
