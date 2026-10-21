"""
Shared Multimodal Cross-Attention Module
Constructs a shared continuous embedding space between S2ST translation representations 
and visual flow-matching lip-sync frame features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMultimodalCrossAttention(nn.Module):
    """
    Shared Multimodal Latent Space Cross-Attention Layer.
    Allows the Lip-Sync flow decoder to query deep semantic representations 
    produced by the streaming speech-to-speech translation model rather than relying 
    solely on noisy output mel-spectrograms.
    """

    def __init__(
        self,
        query_dim: int = 512,      # Visual frame feature dimension
        key_val_dim: int = 512,    # S2ST translation encoder dimension
        out_dim: int = 512,        # Output shared embedding dimension
        num_heads: int = 8,        # Attention heads
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim

        # Queries projection (Visual Face Dynamics / Flow Queries)
        self.q_proj = nn.Linear(query_dim, out_dim)

        # Keys & Values projections (S2ST Semantic Speech Features)
        self.k_proj = nn.Linear(key_val_dim, out_dim)
        self.v_proj = nn.Linear(key_val_dim, out_dim)

        # Multihead Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer Normalization & Feed-Forward Network (FFN)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual_queries: torch.Tensor,     # (Batch, T_video, query_dim)
        speech_semantics: torch.Tensor,   # (Batch, T_speech, key_val_dim)
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes shared multimodal latent representation Z_shared.

        Args:
            visual_queries: Target visual frame features or landmark queries
            speech_semantics: Translated semantic-acoustic features from S2ST backbone
            key_padding_mask: Optional boolean padding mask for speech sequence

        Returns:
            z_shared: (Batch, T_video, out_dim) conditioned multimodal latent space
        """
        # Project inputs into shared subspace
        q = self.q_proj(visual_queries)
        k = self.k_proj(speech_semantics)
        v = self.v_proj(speech_semantics)

        # Cross-Attention: Visual queries key-value attend to translated speech semantics
        attn_out, _ = self.mha(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask
        )

        # Residual connection & Norm 1
        x = self.norm1(q + attn_out)

        # Feed-Forward Network & Norm 2
        z_shared = self.norm2(x + self.ffn(x))

        return z_shared
