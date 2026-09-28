import math
import os
import random

import hydra
import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from catok.models.catok_ddt.vanilla_utils import construct_model
from catok.training.rlds_dataset import DistributedWeightedSampler, make_rlds_dataloader


@hydra.main(
    config_path="../conf/config_robotwin",
    config_name="vanilla_diffusion",
    version_base="1.3",
)
def pretrain(cfg: DictConfig):
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_ddp = cfg.get("use_ddp", False)

    if use_ddp:
        import torch.distributed as dist
        from accelerate import PartialState
        from torch.nn.parallel import DistributedDataParallel as DDP

        distributed_state = PartialState()
        device_id = distributed_state.local_process_index
        torch.cuda.set_device(device_id)
        device = f"cuda:{device_id}"
        is_main_process = distributed_state.is_main_process
        world_size = distributed_state.num_processes
        rank = distributed_state.process_index
    else:
        device_id = 0
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        is_main_process = True
        world_size = 1
        rank = 0

    cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

    vq_mode = cfg["tokenizer"]["vq"]["vq_mode"]
    max_reset_steps = cfg["tokenizer"]["vq"]["max_reset_steps"]
    model = construct_model(cfg["tokenizer"], device=device)
    model.train()
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[device_id],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    save_dir = cfg["ckpt_dir"]
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    data_cfg = cfg["data"]
    action_dim = cfg["tokenizer"]["basic"]["action_dim"]
    per_gpu_batch_size = cfg["train"]["batch_size"] // world_size
    assert cfg["train"]["batch_size"] % world_size == 0

    dataset_path = data_cfg["dataset_path"]
    train_loader_cfg = {
        "dataset_specs": [(dataset_path, data_cfg.get("sampling_weight", 1.0))],
        "data_root": data_cfg["root_dir"],
        "horizon": data_cfg["horizon"],
        "batch_size": per_gpu_batch_size,
        "action_only": data_cfg.get("action_only", True),
        "num_parallel_reads": data_cfg.get("num_parallel_reads", 1),
        "max_episodes": data_cfg.get("max_episodes", None),
        "num_workers": cfg["dataloader_workers"],
        "debug": data_cfg.get("debug", False),
        "target_action_dim": action_dim,
        "stats_path_map": {
            dataset_path: data_cfg["stats_path"],
        },
        "normalizer_method": cfg["normalizer"]["type"],
        "normalizer_config": cfg["normalizer"]["config"],
    }

    if use_ddp:
        train_sampler, train_loader, _ = make_rlds_dataloader(
            **train_loader_cfg,
            distributed=True,
            rank=rank,
            world_size=world_size,
            drop_last=True,
        )
    else:
        train_sampler, train_loader, _ = make_rlds_dataloader(**train_loader_cfg)

    if cfg["train"]["loss_fn"] == "l1":
        loss_fn = nn.L1Loss()
    elif cfg["train"]["loss_fn"] == "l2":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("Unknown loss function.")
    _ = loss_fn

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    num_training_steps = cfg["train"]["epochs"] * len(train_loader)
    warmup = cfg["train"].get("warmup", 0)
    warmup_steps = int(num_training_steps * warmup / 100.0) if warmup > 0 else 0

    def combine_scheduler(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        cosine_steps = num_training_steps - warmup_steps
        step_in_cosine = current_step - warmup_steps
        return 0.5 * (1 + math.cos(math.pi * step_in_cosine / cosine_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=combine_scheduler,
    )

    exp_name = cfg["wandb_expname"]
    if is_main_process and cfg.get("use_wandb", False):
        wandb.init(name=exp_name, project=cfg["wandb_project"], config=cfg)

    epochs = cfg["train"]["epochs"]
    if is_main_process:
        print("DataLoader Len:", len(train_loader))

    steps = 0
    for epoch in tqdm.trange(epochs, disable=(not is_main_process)):
        if isinstance(train_sampler, DistributedWeightedSampler):
            train_sampler.set_epoch(epoch)

        for batch in tqdm.tqdm(train_loader, disable=(not is_main_process)):
            steps += 1
            optimizer.zero_grad()
            batch = batch.to(device=device, dtype=torch.float32)
            recon_x, loss, loss_dict, indices, perplexity = model(batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if vq_mode in ["RVQ_EMA", "VQ_EMA"] and steps == max_reset_steps:
                if use_ddp:
                    model.module.vq.set_threshold(0)
                else:
                    model.vq.set_threshold(0)

            if is_main_process and cfg.get("use_wandb", False) and steps % cfg.get("log_every_n_steps", 10) == 0:
                wandb.log({"recon_loss": loss_dict["recon"]}, step=steps)
                if vq_mode == "RVQ_EMA":
                    wandb.log({"vq_loss": loss_dict["vq_loss"]}, step=steps)
                    for i in range(3):
                        wandb.log(
                            {
                                f"active_code_{i}": len(torch.unique(indices[i])),
                                f"perplexity_{i}": perplexity[i],
                            },
                            step=steps,
                        )
                elif vq_mode is not None:
                    wandb.log(
                        {
                            "vq_loss": loss_dict["vq_loss"],
                            "active_code": len(torch.unique(indices)),
                            "perplexity": perplexity,
                        },
                        step=steps,
                    )
                if cfg["train"]["dct_loss"]:
                    wandb.log({"dct_loss": loss_dict["dct"]}, step=steps)

            if is_main_process and steps % cfg["save_every_n_steps"] == 0:
                model_state = model.module.state_dict() if use_ddp else model.state_dict()
                save_path = os.path.join(save_dir, f"{steps:06d}.pth")
                torch.save(
                    {
                        "model_state_dict": model_state,
                        "config": cfg,
                    },
                    save_path,
                )

    if is_main_process and cfg.get("use_wandb", False):
        wandb.finish()

    if use_ddp:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    pretrain()
