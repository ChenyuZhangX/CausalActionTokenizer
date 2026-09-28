import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings**0.5, 1.0 / num_embeddings**0.5)

    def forward(self, inputs):
        # inputs: (Batch, Seq_len, Dim)
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # get nearest codes
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # quantize
        quantized = torch.matmul(encodings, self.embedding.weight).reshape(inputs.shape)
        
        # VQ loss + Commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # calc Perplexity (rate of Codebook usage)
        avg_probs = torch.mean(encodings, dim=0)    # code被选择的概率
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, encoding_indices.reshape(inputs.shape[:-1]), perplexity
    
    def decode(self, indices):
        """
        encoding_indices: (Batch, Seq_len)
        return: (Batch, Seq_len, Dim)
        """
        return self.embedding(indices)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, vq_dim=16, commitment_cost=0.25, decay=0.99, epsilon=1e-5, threshold=1.0, check_every=100, add_vq_latent=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        if add_vq_latent:
            self.vq_dim = vq_dim
        else:
            self.vq_dim = embedding_dim

        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        if add_vq_latent:
            self.pre_proj = nn.Linear(self.embedding_dim, self.vq_dim)
            self.post_proj = nn.Linear(self.vq_dim, self.embedding_dim)
        else: 
            self.pre_proj = nn.Identity()
            self.post_proj = nn.Identity()
        
        # EMA
        self.decay = decay
        self.epsilon = epsilon

        # dead code threshold (times of usage)
        self.threshold = threshold
        self.check_every = check_every
        self._total_steps = 0
        
        # init Embedding (register_buffer)
        embedding = torch.randn(self.num_embeddings, self.vq_dim)
        self.register_buffer('embedding', embedding)
        
        # auxiliary variable for EMA
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', embedding.clone())

    def expire_codes(self, inputs):
        """
        Restart dead codes.
        inputs: usually flat_input passed in via forward(), used for new code
        """
        if not self.training:
            return

        # find indices of dead codes
        dead_indices = torch.where(self.ema_cluster_size < self.threshold)[0]
        
        if len(dead_indices) > 0:
            # sample randomly from inputs
            n_dead = len(dead_indices)
            flat_input = inputs.reshape(-1, self.vq_dim)
            n_input = flat_input.shape[0]
            perm = torch.randperm(n_input, device=inputs.device)
            # (If there are not enough input samples, reuse them.)
            idx = perm[:n_dead] if n_input >= n_dead else perm.repeat(int(n_dead/n_input)+1)[:n_dead]
            new_embeddings = flat_input[idx]

            # update dead codes, ema_cluster_size, ema_w
            self.embedding.data[dead_indices] = new_embeddings
            self.ema_w.data[dead_indices] = new_embeddings
            self.ema_cluster_size.data[dead_indices] = self.threshold

            # print('Expired Codes: ', n_dead)
    
    def forward(self, inputs):
        # inputs: (Batch, Seq_len, Dim) -> (Batch * Seq_len, Dim)
        
        z_e = self.pre_proj(inputs)

        flat_input = z_e.reshape(-1, self.vq_dim)
        
        # get neareast codes
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (N, 1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # (N, Num_Embeddings)
        
        # EMA update
        if self.training:
            self._total_steps += 1

            # get code-use times in batch
            encodings_sum = encodings.sum(0)
            # (Num_Embeddings, N) @ (N, Dim) -> (Num_Embeddings, Dim)
            dw = torch.matmul(encodings.t(), flat_input)

            # sync across GPUs if using DDP
            if dist.is_initialized():
                dist.all_reduce(encodings_sum)
                dist.all_reduce(dw)

            # update cluster size
            self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # Laplace Smoothing to avoid being dividied by 0
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            
            # update Codebook
            self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

            # restart dead codes
            if self.check_every != 0 and self._total_steps % self.check_every == 0:
                self.expire_codes(z_e)
        
        # quantize
        quantized = torch.matmul(encodings, self.embedding).reshape(z_e.shape)
        
        # only Commitment Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = z_e + (quantized - z_e).detach()

        quantized_out = self.post_proj(quantized)
        
        # calc Perplexity (rate of Codebook usage)
        avg_probs = torch.mean(encodings, dim=0)    # code被选择的概率
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized_out, encoding_indices.reshape(inputs.shape[:-1]), perplexity
    
    def set_threshold(self, threshold = 0):
        self.threshold = threshold

    def decode(self, indices):
        """
        encoding_indices: (Batch, Seq_len)
        return: (Batch, Seq_len, Dim)
        """
        quantized = F.embedding(indices, self.embedding)
        return self.post_proj(quantized)

class ResidualVQEMA(nn.Module):
    def __init__(self, num_quantizers, num_embeddings, embedding_dim, vq_dim=16, commitment_cost=0.25, decay=0.99, threshold=1.0, check_every=100, add_vq_latent=True):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.vq_dim = vq_dim if add_vq_latent else embedding_dim

        if add_vq_latent:
            self.pre_proj = nn.Linear(embedding_dim, self.vq_dim)
            self.post_proj = nn.Linear(self.vq_dim, embedding_dim)
        else:
            self.pre_proj = nn.Identity()
            self.post_proj = nn.Identity()

        self.layers = nn.ModuleList([
            VectorQuantizerEMA(
                num_embeddings=num_embeddings,
                embedding_dim=self.vq_dim,
                commitment_cost=commitment_cost,
                decay=decay,
                threshold=threshold,
                check_every=check_every,
                add_vq_latent=False,
            )
            for _ in range(num_quantizers)
        ])

    def forward(self, inputs):
        # inputs: (Batch, Seq, Dim)
        z_e = self.pre_proj(inputs)

        quantized_out = 0.0
        residual = z_e

        all_losses = 0.0
        all_perplexities = []
        all_indices = []

        for layer in self.layers:
            loss, quantized, indices, perplexity = layer(residual)

            quantized_out = quantized_out + quantized
            residual = z_e - quantized_out

            all_losses += loss
            all_perplexities.append(perplexity)
            all_indices.append(indices)

        return all_losses, self.post_proj(quantized_out), all_indices, all_perplexities

    def set_threshold(self, threshold = 0):
        for layer in self.layers:
            layer.threshold = threshold

    def decode(self, all_indices):
        """
        encoding_indices: (Batch, N_Codebook, Seq_len)
        return: (Batch, Seq_len, Dim)
        """
        quantized_out = 0.0

        for i, layer in enumerate(self.layers):
            quantized_out = quantized_out + layer.decode(all_indices[:, i])

        return self.post_proj(quantized_out)