import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

DEFAULT_DATASET_NAME_ALIASES = {
    'bridge': 'bridge',
    'libero': 'Libero_RLDS/libero_10_no_noops',
    'Libero_RLDS/libero_10_no_noops': 'Libero_RLDS/libero_10_no_noops',
}


def _resolve_dataset_name(dataset_name: str) -> str:
    return DEFAULT_DATASET_NAME_ALIASES.get(dataset_name, dataset_name)


def _build_default_datasets_cfg(data_root: str):
    bridge_stats = str(Path(data_root) / 'bridge' / 'stats.json')
    return {
        'bridge': {
            'weight': 1.0,
            'stats_path': bridge_stats,
            'normalizer_method': 'bridge',
            'action_dim': 7,
        },
        'Libero_RLDS/libero_spatial_no_noops': {
            'weight': 1.0,
            'stats_path': str(Path(data_root) / 'Libero_RLDS' / 'libero_spatial_no_noops' / 'stats.json'),
            'action_dim': 7,
        },
        'Libero_RLDS/libero_goal_no_noops': {
            'weight': 1.0,
            'stats_path': str(Path(data_root) / 'Libero_RLDS' / 'libero_goal_no_noops' / 'stats.json'),
            'action_dim': 7,
        },
        'Libero_RLDS/libero_object_no_noops': {
            'weight': 1.0,
            'stats_path': str(Path(data_root) / 'Libero_RLDS' / 'libero_object_no_noops' / 'stats.json'),
            'action_dim': 7,
        },
        'Libero_RLDS/libero_10_no_noops': {
            'weight': 1.0,
            'stats_path': str(Path(data_root) / 'Libero_RLDS' / 'libero_10_no_noops' / 'stats.json'),
            'action_dim': 7,
        },
    }


def _dct(x):
    # x: (N, T, D)
    # return the DCT of x
    # the DCT is a linear transformation
    # the DCT is a linear transformation
    x = x.permute(0, 2, 1)
    x_dct = dct(x, norm="ortho")
    return x_dct.permute(0, 2, 1)

def calc_loss(recon_x, x, dct_loss=False, loss_fn=F.mse_loss, padding_mask=None):
    """
    Args:
        padding_mask: bool tensor (B, H, A), True = valid position, False = padded.
                      When provided, reconstruction loss is computed only on valid positions.
    """
    if padding_mask is not None:
        loss = loss_fn(recon_x[padding_mask], x[padding_mask])
    else:
        loss = loss_fn(recon_x, x)
    loss_dict = {"recon": loss.item()}

    if dct_loss:
        dct_l = F.mse_loss(_dct(x), _dct(recon_x))
        loss += dct_l
        loss_dict["dct"] = dct_l.item()

    return loss, loss_dict

def get_mask(t, n_tokens: int=5, is_causal: bool=True):
    if not is_causal:
        mask = torch.ones(t.shape[0], n_tokens, dtype=torch.bool, device=t.device)
        return mask
    
    t_thresholds = t.unsqueeze(1) * n_tokens # (b, 1)
    indices = torch.arange(1, n_tokens+1).float().to(t.device)
    mask = indices > t_thresholds
    return mask

def visualize_attention_mask(mask, hidden_dim=10):
    """
    可视化 Hidden-Action Attention Mask (Hidden在前, Action在后)
    
    参数:
    - mask: 2D numpy array 或 torch tensor
    - hidden_dim: hidden states 的维度，默认为 10
    """
    # 确保输入是 numpy 数组
    if hasattr(mask, 'numpy'):
        mask = mask.numpy()
    
    total_dim = mask.shape[0]
    action_dim = total_dim - hidden_dim

    cmap = mcolors.ListedColormap(['#F0F0F0', '#27AE60'])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

    # 3. 设置坐标轴标签 (Hidden 在前, Action 在后)
    labels = [f'H{i+1}' for i in range(hidden_dim)] + [f'A{i+1}' for i in range(action_dim)]
    ticks = np.arange(total_dim)
    
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    # 4. 绘制虚线分割线
    # 现在的分割点在第 hidden_dim 个元素之后
    divider = hidden_dim - 0.5
    
    line_params = {
        'color': '#333333',
        'linestyle': '--', # 简单虚线
        'linewidth': 2,
        'alpha': 0.8       # 稍微透明一点，不遮挡背景
    }
    
    ax.axhline(y=divider, **line_params)
    ax.axvline(x=divider, **line_params)

    # 5. 添加图例
    legend_elements = [
        Patch(facecolor='#27AE60', label='Attend (True)'),
        Patch(facecolor='#F0F0F0', edgecolor='#CCCCCC', label='Masked (False)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.title(f"Attention Mask: Hidden Encoder States ({hidden_dim}) & Action ({action_dim})", pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig('tmp.png', dpi=300, bbox_inches='tight')

def construct_model(model_config, device):
    from catok.models.catok_ddt.vanilla_tokenizer import VanillaVQDiffusion
    
    basic_cfg = model_config.get('basic', {})
    encoder_cfg = model_config['encoder']
    embedder_cfg = encoder_cfg.get('embedder', {})
    vq_cfg = model_config['vq']
    decoder_cfg = model_config.get('decoder', {})

    add_vq_latent = encoder_cfg.get('add_vq_latent', True)
    d_vq = encoder_cfg.get('d_vq', 16)
    
    decoder_kwargs = dict(decoder_cfg)
    context_see_xt = decoder_kwargs.pop('context_see_xt', True)
    noise_level = decoder_kwargs.pop('noise_level', 1.0)
    flow_cfg = dict(model_config.get('flow', {}))
    overlap_cfg = model_config.get('overlap_infonce', {}) or {}
    adjacent_cfg = model_config.get('adjacent_infonce', {}) or {}
    or_direct_cfg = model_config.get('or_direct', {}) or {}

    model = VanillaVQDiffusion(
        horizon=basic_cfg['horizon'],
        action_dim=basic_cfg['action_dim'],
        is_causal=basic_cfg.get('is_causal', True),
        add_noise=basic_cfg.get('add_noise', False),
        flow_type=basic_cfg.get('flow_type', 'vanilla'),
        latent_len=encoder_cfg['num_tokens'],
        # encoder
        d_model=encoder_cfg['d_model'],
        num_layers=encoder_cfg['num_layers'],
        encoder_mode=encoder_cfg['encoder_mode'],
        add_vq_latent=add_vq_latent,
        **embedder_cfg,
        # vq
        vq_mode=vq_cfg['vq_mode'],
        d_vq=d_vq,
        n_e=vq_cfg['codebook_size'],
        vq_check_every=vq_cfg['check_every'],
        vq_threshold=vq_cfg['threshold'],
        # decoder
        context_see_xt=context_see_xt,
        noise_level=noise_level,
        flow_cfg=flow_cfg,
        overlap_infonce_weight=float(overlap_cfg.get('weight', 0.0)),
        overlap_infonce_std=float(overlap_cfg.get('noise_std', 0.02)),
        overlap_infonce_temperature=float(overlap_cfg.get('temperature', 0.07)),
        adjacent_infonce_weight=float(adjacent_cfg.get('weight', 0.0)),
        adjacent_infonce_temperature=float(adjacent_cfg.get('temperature', 0.07)),
        or_direct_weight=float(or_direct_cfg.get('weight', 0.0)),
        or_direct_temperature=float(or_direct_cfg.get('temperature', 0.1)),
        **decoder_kwargs,
    ).to(device)
    
    return model

def load_checkpoint(model_path, device, eval_mode=False):
    checkpoint = torch.load(model_path, map_location=device)

    # if eval_mode:
    #     checkpoint['config']['tokenizer']['decoder']['noise_level'] = 1.0

    model = construct_model(checkpoint['config']['tokenizer'], device)
    result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if result.missing_keys:
        print("\n[Missing Keys]: These parameters exist in the MODEL but not in the CKPT:")
        for key in result.missing_keys:
            print(f"  - {key}")
    if result.unexpected_keys:
        print("\n[Unexpected Keys]: These parameters exist in the CKPT but not in the MODEL:")
        for key in result.unexpected_keys:
            print(f"  - {key}")

    return model, checkpoint['config']


def construct_dataloader(
    cfg,
    batch_size=None,
    num_workers=None,
    distributed=False,
    rank=None,
    world_size=None,
    drop_last=False,
    dataset_names=None,
    eval_mode=False,
):
    """从完整 cfg 中自动解析参数并创建 multi-dataset DataLoader。

    Args:
        cfg: 完整配置字典，包含 data, tokenizer, normalizer 等字段。
        batch_size: 覆盖 cfg 中的 batch_size，None 则使用 cfg['train']['batch_size']。
        num_workers: 覆盖 cfg 中的 dataloader_workers，None 则使用 cfg 中的值。
        distributed: 是否 DDP 模式。
        rank: DDP rank，None 则自动获取。
        world_size: DDP world_size，None 则自动获取。
        drop_last: 是否 drop_last。
        dataset_names: 要加载的数据集名称列表，None 则加载 cfg 中所有数据集。
            可传入单个字符串或字符串列表，如 'bridge' 或 ['bridge', 'Libero_RLDS/libero_10_no_noops']。

    Returns:
        (sampler, loader, concat_ds)
    """
    from catok.training.rlds_dataset import make_rlds_dataloader

    data_cfg = cfg['data']
    action_dim = cfg['tokenizer']['basic']['action_dim']
    normalizer_cfg = cfg.get('normalizer', {})
    default_normalizer_method = normalizer_cfg.get('type', 'qq')
    normalizer_config = normalizer_cfg.get('config', {})

    # 从 data.datasets 解析 per-dataset 参数
    data_root = data_cfg.get('root_dir', 'data')
    datasets_cfg = data_cfg.get('datasets', {})
    if not datasets_cfg:
        datasets_cfg = _build_default_datasets_cfg(data_root)
        print(
            "[construct_dataloader] cfg.data.datasets is empty. "
            "Using default datasets: bridge:1, libero:5."
        )

    # 过滤数据集
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        dataset_names = [_resolve_dataset_name(name) for name in dataset_names]
        datasets_cfg = {k: v for k, v in datasets_cfg.items() if k in dataset_names}

    if not datasets_cfg:
        available_datasets = list(data_cfg.get('datasets', {}).keys()) or list(
            _build_default_datasets_cfg(data_root).keys()
        )
        raise ValueError(
            f"No dataset matched dataset_names={dataset_names}. "
            f"Available datasets: {available_datasets}"
        )

    dataset_specs = []
    stats_path_map = {}
    normalizer_method_map = {}
    action_dim_map = {}
    horizon_map = {}

    for ds_path, ds_cfg in datasets_cfg.items():
        dataset_specs.append((ds_path, ds_cfg['weight']))
        if 'stats_path' in ds_cfg:
            stats_path_map[ds_path] = ds_cfg['stats_path']
        if 'normalizer_method' in ds_cfg:
            normalizer_method_map[ds_path] = ds_cfg['normalizer_method']
        if 'action_dim' in ds_cfg:
            action_dim_map[ds_path] = ds_cfg['action_dim']
        if 'horizon' in ds_cfg:
            horizon_map[ds_path] = ds_cfg['horizon']

    target_horizon = data_cfg['horizon']
    _batch_size = batch_size if batch_size is not None else cfg.get('train', {}).get('batch_size', 64)
    _num_workers = num_workers if num_workers is not None else cfg.get('dataloader_workers', 0)

    # build action_start_map from per-dataset config
    action_start_map = {}
    for ds_path, ds_cfg in datasets_cfg.items():
        if 'action_start' in ds_cfg:
            action_start_map[ds_path] = ds_cfg['action_start']

    loader_cfg = {
        'dataset_specs': dataset_specs,
        'data_root': data_root,
        'horizon': target_horizon,
        'batch_size': _batch_size,
        'action_only': True,
        'num_parallel_reads': 8,
        'num_workers': _num_workers,
        'target_action_dim': action_dim,
        'target_horizon': target_horizon if horizon_map else None,
        'stats_path_map': stats_path_map,
        'normalizer_method': default_normalizer_method,
        'normalizer_method_map': normalizer_method_map,
        'normalizer_config': normalizer_config,
        'action_dim_map': action_dim_map or None,
        'horizon_map': horizon_map or None,
        'action_start_map': action_start_map or None,
    }

    return make_rlds_dataloader(
        **loader_cfg,
        eval_mode=eval_mode,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        drop_last=drop_last,
    )


def compare_encoder_params(ckpt_path1, ckpt_path2, prefix='encoder.', device='cpu'):
    """比较两个 checkpoint 中 encoder 参数是否发生变化。

    Args:
        ckpt_path1: 第一个 checkpoint 路径
        ckpt_path2: 第二个 checkpoint 路径
        prefix:     要比较的参数前缀，默认 'encoder.'
        device:     加载 checkpoint 的设备
    """
    sd1 = torch.load(ckpt_path1, map_location=device)['model_state_dict']
    sd2 = torch.load(ckpt_path2, map_location=device)['model_state_dict']

    keys1 = {k for k in sd1 if k.startswith(prefix)}
    keys2 = {k for k in sd2 if k.startswith(prefix)}

    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2

    if only_in_1:
        print(f"[Only in ckpt1] {len(only_in_1)} keys:")
        for k in sorted(only_in_1):
            print(f"  {k}")
    if only_in_2:
        print(f"[Only in ckpt2] {len(only_in_2)} keys:")
        for k in sorted(only_in_2):
            print(f"  {k}")

    changed, unchanged = [], []
    for k in sorted(common):
        if not torch.equal(sd1[k], sd2[k]):
            # diff = (sd1[k].float() - sd2[k].float()).abs()
            changed.append(k)
        else:
            unchanged.append(k)

    print(f"\n[Summary] prefix='{prefix}'")
    print(f"  Total keys compared : {len(common)}")
    print(f"  Unchanged           : {len(unchanged)}")
    print(f"  Changed             : {len(changed)}")

    if changed:
        print("\n[Changed parameters] (max_diff, mean_diff):")
        for k in unchanged:
            print(f"  {k:60s}")
    else:
        print("\nAll encoder parameters are identical between the two checkpoints.")
