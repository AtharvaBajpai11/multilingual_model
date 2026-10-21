"""
DiFlowDubber Diffusion / Flow-Matching Lip-Sync Decoder Backbone
Implements Rectified Flow / Optimal Transport Flow Matching (OT-Flow) for conditioned face video frame synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiFlowDubberDecoder(nn.Module):
    """
    Diffusion / Flow-Matching Lip-Sync Decoder (DiFlowDubber Lineage).
    Predicts the conditional velocity field v_t(x_t | Z_shared) to generate 
    synchronized target lip movement & face motion heatmaps from continuous flow.
    """

    def __init__(
        self,
        frame_channels: int = 3,       # RGB channels or latent face feature dim per frame
        hidden_dim: int = 512,         # Model backbone feature dimension
        cond_dim: int = 512,           # Dimension of conditioned Z_shared multimodal latent
        num_blocks: int = 4,           # Number of residual flow blocks
        spatial_size: int = 64,        # Target face crop resolution (64x64)
    ):
        super().__init__()
        self.frame_channels = frame_channels
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.spatial_size = spatial_size

        # Timestep embedding MLP for t in [0, 1]
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Reference Face Image & Noisy Frame Input Conv
        self.input_conv = nn.Conv2d(frame_channels * 2, hidden_dim, kernel_size=3, padding=1)

        # Multimodal Latent Condition Adapter
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # Flow residual blocks with cross-attention / feature modulation
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "conv1": nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                "norm1": nn.GroupNorm(8, hidden_dim),
                "conv2": nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                "norm2": nn.GroupNorm(8, hidden_dim),
                "time_proj": nn.Linear(hidden_dim, hidden_dim),
                "cond_attn": nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True),
            })
            for _ in range(num_blocks)
        ])

        # Velocity output head v_t(x_t, t, Z_shared)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, frame_channels, kernel_size=3, padding=1)
        )

    def forward(
        self,
        x_t: torch.Tensor,              # Noisy/interpolated face frames: (Batch*Time, C, H, W)
        t: torch.Tensor,                # Continuous timesteps in [0, 1]: (Batch*Time,)
        ref_face: torch.Tensor,         # Reference identity face frame: (Batch*Time, C, H, W)
        z_shared: torch.Tensor,         # IsoSync conditioned latent: (Batch*Time, Cond_Dim)
    ) -> torch.Tensor:
        """
        Predicts conditional velocity field v_t for flow matching ODE solver.

        Args:
            x_t: (B*T, C, H, W) current noisy image state at time t
            t: (B*T,) timestep scalar tensor
            ref_face: (B*T, C, H, W) reference un-dubbed face frame
            z_shared: (B*T, cond_dim) conditioned feature vector from IsoSync-Bridge

        Returns:
            v_t: (B*T, C, H, W) predicted velocity field
        """
        # Embed timesteps
        t_emb = self.time_mlp(t.unsqueeze(-1)) # (B*T, hidden_dim)

        # Embed latent condition
        cond_emb = self.cond_proj(z_shared)     # (B*T, hidden_dim)

        # Concatenate noisy image with reference identity frame
        x_in = torch.cat([x_t, ref_face], dim=1) # (B*T, 2*C, H, W)
        h = self.input_conv(x_in)

        # Apply residual flow blocks
        for block in self.blocks:
            residual = h
            h = block["conv1"](F.silu(block["norm1"](h)))
            
            # Add time embedding
            t_feat = block["time_proj"](t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + t_feat
            
            h = block["conv2"](F.silu(block["norm2"](h)))

            # Cross-attend to Z_shared multimodal condition
            B_T, C, H, W = h.shape
            spatial_tokens = h.flatten(2).transpose(1, 2) # (B*T, H*W, C)
            cond_tokens = cond_emb.unsqueeze(1)          # (B*T, 1, C)
            
            attn_out, _ = block["cond_attn"](spatial_tokens, cond_tokens, cond_tokens)
            h = residual + attn_out.transpose(1, 2).view(B_T, C, H, W)

        velocity_field = self.out_conv(h)
        return velocity_field

    @torch.no_grad()
    def sample_ode(
        self,
        ref_face: torch.Tensor,
        z_shared: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        r"""
        Solves OT-Flow ODE x_1 = x_0 + \int_0^1 v_t(x_t, t, Z_shared) dt using Euler integration.
        """
        device = ref_face.device
        BT, C, H, W = ref_face.shape

        # Initial standard Gaussian noise x_0 ~ N(0, I)
        x_t = torch.randn(BT, self.frame_channels, H, W, device=device)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((BT,), t_val, device=device, dtype=torch.float32)
            v_t = self.forward(x_t, t_tensor, ref_face, z_shared)
            x_t = x_t + v_t * dt

        return torch.clamp(x_t, -1.0, 1.0)
