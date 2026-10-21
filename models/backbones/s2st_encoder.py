"""
Streaming S2ST Encoder Backbone Wrapper
Supports streaming acoustic and semantic feature extraction from input speech audio.
Compatible with SeamlessM4T / w2v-BERT / Conformer lineage architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingS2STEncoder(nn.Module):
    """
    Pretrained Streaming S2ST Speech Encoder Backbone.
    Extracts frame-level semantic-acoustic representations from input audio.
    Supports streaming causal attention masking and feature projection.
    """

    def __init__(
        self,
        input_dim: int = 80,       # Mel-spectrogram bins or raw audio frame feature dim
        hidden_dim: int = 512,     # Encoder hidden dimension
        num_layers: int = 6,       # Number of conformer / transformer layers
        num_heads: int = 8,        # Multi-head attention heads
        dropout: float = 0.1,
        streaming: bool = True,    # Enables causal masking for real-time streaming
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.streaming = streaming

        # Input Convolutional Subsampling (4x temporal subsampling)
        self.subsampling = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, hidden_dim),
        )

        # Causal / Non-causal Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear projection for downstream target alignment
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates an upper-triangular causal mask for streaming inference."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
        self,
        audio_features: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of S2ST Speech Encoder.

        Args:
            audio_features: (Batch, Channels/Mel_Bins, Time_Frames) or (Batch, Time_Frames, Mel_Bins)
            padding_mask: (Batch, Time_Frames) optional boolean mask

        Returns:
            latent_features: (Batch, Subsampled_Time, hidden_dim)
        """
        # Ensure shape is (Batch, Mel_Bins, Time_Frames) for 1D convolution
        if audio_features.dim() == 3 and audio_features.shape[1] != self.input_dim:
            if audio_features.shape[2] == self.input_dim:
                audio_features = audio_features.transpose(1, 2)

        # 4x Convolutional Subsampling
        x = self.subsampling(audio_features) # (Batch, hidden_dim, Time_Subsampled)
        x = x.transpose(1, 2)                # (Batch, Time_Subsampled, hidden_dim)

        B, T, D = x.shape
        attn_mask = None
        if self.streaming:
            attn_mask = self._generate_causal_mask(T, x.device)

        # Transformer Encoder pass
        out = self.encoder(x, mask=attn_mask)
        latent_features = self.proj_out(out)

        return latent_features
