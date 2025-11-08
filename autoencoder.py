import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, feature_dim=20, hidden_dim=256, codebook_size=2048, code_dim=256, beta=0.25):
        super().__init__()
        self.beta = beta
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        
        # --- New Projection Layer ---
        # Projects the feature vector (e.g., 20) to the hidden dim (e.g., 256)
        self.projection = nn.Linear(feature_dim, hidden_dim)

        # Encoder: 1D CNN downsampling
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pre_quant = nn.Conv1d(hidden_dim, code_dim, kernel_size=1)
        
        # Codebook (learnable embeddings)
        self.codebook = nn.Parameter(torch.randn(codebook_size, code_dim))
        
        # Decoder: Symmetric upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(code_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        
        # --- New Output Layer ---
        # Projects back from hidden_dim to the original feature_dim
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

        # EMA tracking (for codebook updates)
        self.register_buffer('ema_count', torch.ones(codebook_size))
        self.register_buffer('ema_weight', self.codebook.clone())

    def forward(self, x):
        # x: [B, F, L] (e.g., [32, 20, 512])
        
        # --- 1. Project Features ---
        x = x.permute(0, 2, 1)  # [B, L, F]
        x = self.projection(x)  # [B, L, H]
        x = x.permute(0, 2, 1)  # [B, H, L]
        
        # --- 2. Encode ---
        z = self.encoder(x)
        z = self.pre_quant(z)  # [B, code_dim, L_down] (e.g., [32, 256, 64])
        
        # --- 3. Quantization ---
        z_flat = z.permute(0, 2, 1).reshape(-1, self.code_dim)
        dist = torch.cdist(z_flat, self.codebook)
        indices = dist.argmin(dim=1)
        z_q_flat = self.codebook[indices]
        z_q = z_q_flat.view(z.shape[0], z.shape[2], self.code_dim).permute(0, 2, 1)
        
        # --- 4. Straight-through estimator ---
        z_q = z + (z_q - z).detach()
        
        # --- 5. Decode ---
        recon = self.decoder(z_q)  # [B, H, L]
        
        # --- 6. Project back to Features ---
        recon = recon.permute(0, 2, 1) # [B, L, H]
        recon = self.output_layer(recon) # [B, L, F]
        recon = recon.permute(0, 2, 1) # [B, F, L]
        
        return recon, z, z_q, indices

    def loss(self, x, recon, z, z_q):
        recon_loss = F.mse_loss(recon, x)
        commit_loss = F.mse_loss(z_q.detach(), z) * self.beta
        return recon_loss + commit_loss, recon_loss, commit_loss

    def update_codebook(self, z_flat, indices, decay=0.99):
        # EMA updates
        onehot = F.one_hot(indices, self.codebook_size).float()
        m = onehot.sum(0)
        self.ema_count = decay * self.ema_count + (1 - decay) * m
        n = onehot.T @ z_flat
        self.ema_weight = decay * self.ema_weight + (1 - decay) * n
        
        # Laplace smoothing to avoid division by zero
        n_total = self.ema_count.sum()
        self.codebook.data = (self.ema_weight + 1e-5) / (self.ema_count.unsqueeze(1) + 1e-5 * n_total)