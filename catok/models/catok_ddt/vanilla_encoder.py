import torch
import torch.nn as nn

from catok.models.catok_ddt.modules import DualBlock


class LinearEmbedder(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
    def forward(self, x):
        return self.proj(x)

class PatchEmbedder(nn.Module):
    def __init__(self, horizon, action_dim, channels_h, channels_a, d_model, use_weight_norm=True):
        super().__init__()

        patch = (horizon//channels_h, action_dim//channels_a)
        padding = (
            ((patch[0] - (horizon % patch[0])) % patch[0]) // 2,
            ((patch[1] - (action_dim % patch[1])) % patch[1]) // 2,
        )
        
        self.patch = patch
        self.channels_h = channels_h
        self.channels_a = channels_a
        self.padding = padding

        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch,
            stride=patch,
            padding=padding
        )

        if use_weight_norm:
            self.proj = nn.utils.weight_norm(self.proj)

        out_h = (horizon + 2 * padding[0] - patch[0]) // patch[0] + 1
        out_a = (action_dim + 2 * padding[1] - patch[1]) // patch[1] + 1
        self.num_patches_h = out_h
        self.num_patches_a = out_a
        self.num_patches = out_h * out_a

    def forward(self, x):
        #  (B, H, A) -> (B, C_H*C_A, d_model)
        x = x.unsqueeze(1)  # (B, 1, H, A)
        # print("after unsqueeze:", x.size())
        x = self.proj(x)    # (B, d_model, C_H, C_A)
        x = x.flatten(2)    # (B, d_model, N)
        x = x.transpose(1, 2)   # (B, N, d_model)

        return x
    
class VanillaEncoder(nn.Module):
    def __init__(
        self,
        horizon=20,
        action_dim=8,
        d_model=128,
        num_layers=4,
        encoder_mode='Transformer', 
        embedder_type='linear',
        # add_vq_latent=True,
        **kwargs
    ):
        super().__init__()
        
        # Embedder
        if embedder_type == 'linear':
            max_seq_len = horizon
            self.embedder = LinearEmbedder(action_dim, d_model)
        elif embedder_type == 'patch':
            channels_h = kwargs.get('channels_h', 1)
            channels_a = kwargs.get('channels_a', 8)
            self.embedder = PatchEmbedder(
                horizon=horizon,
                action_dim=action_dim,
                channels_h=channels_h,
                channels_a=channels_a,
                d_model=d_model,
                use_weight_norm=kwargs.get('use_weight_norm', True)
            )
            max_seq_len = self.embedder.num_patches
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")

        # positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # encoder layers
        self.encoder_mode = encoder_mode
        if encoder_mode == 'Transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=kwargs.get('nhead', 4), 
                dim_feedforward=kwargs.get('dim_feedforward', 4*d_model), 
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)     
        elif encoder_mode == 'Dual':
            # Query Tokens
            latent_len = kwargs.get('latent_len', None)
            assert latent_len is not None, "Dual mode requires latent_len"
            self.query_tokens = nn.Parameter(torch.randn(1, latent_len, d_model) * 0.02)

            self.encoder = nn.ModuleList([
                DualBlock(
                    hidden_size=d_model,
                    num_heads=kwargs.get('nhead', 4),
                    mlp_ratio=kwargs.get('mlp_ratio', 4),
                    query_dim=d_model,
                    bidirectional=False,
                    time_adaln=True,
                    qk_norm=False,
                    diti=None,
                ) for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unsupported encoder_mode: {encoder_mode}")

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Input_Dim]
        """
        x = self.embedder(x)
        if x.shape[1] != self.pos_embedding.shape[1]:
            # Keep forward robust to config changes in patch layouts.
            pos = self.pos_embedding.transpose(1, 2)
            pos = torch.nn.functional.interpolate(
                pos, size=x.shape[1], mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos_embedding
        x = x + pos

        if self.encoder_mode == 'Transformer':
            out = self.encoder(x)
        elif self.encoder_mode == 'Dual':
            b = x.shape[0]
            query_tokens = self.query_tokens.expand(b, -1, -1)
            
            for block in self.encoder:
                x, query_tokens = block(x, query_tokens)
            out = query_tokens

        return out