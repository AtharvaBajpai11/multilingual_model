"""
IsoSyncBridge: The Novel Connecting Module
Sits between the Pretrained Streaming S2ST backbone and the Pretrained Flow-Matching Lip-Sync Decoder.
Enables joint duration control and shared multimodal cross-attention conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .duration_predictor import IsochronousDurationPredictor
from .cross_modal_attn import SharedMultimodalCrossAttention


class IsoSyncBridge(nn.Module):
    """
    IsoSync-Bridge: Isochronous Multimodal Latent Bridge Module.
    
    Functions:
    1. Extracts temporal motion envelope from source face video frames.
    2. Predicts target acoustic/phoneme duration and expands features via Length Regulation.
    3. Fuses speech semantic vectors with visual queries in a Shared Multimodal Latent Space (Z_shared).
    4. Projects Z_shared into velocity guidance vectors for the flow-matching lip-sync decoder.
    """

    def __init__(
        self,
        s2st_dim: int = 512,       # Dimension of S2ST encoder representation
        flow_dim: int = 512,       # Dimension expected by DiFlowDubber decoder
        video_dim: int = 64,       # Dimension of video temporal envelope features
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.s2st_dim = s2st_dim
        self.flow_dim = flow_dim
        self.video_dim = video_dim

        # Source Video Motion Encoder (extracts temporal envelope from facial frame differences)
        self.video_envelope_encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, video_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128), # Standardized length projection
        )

        # 1. Isochronous Duration Predictor (Novel Component)
        self.duration_predictor = IsochronousDurationPredictor(
            speech_dim=s2st_dim,
            video_dim=video_dim,
            hidden_dim=256,
            dropout=dropout,
        )

        # Visual query projection for target video frame rate alignment
        self.visual_query_gen = nn.Sequential(
            nn.Linear(video_dim, flow_dim),
            nn.GELU(),
            nn.Linear(flow_dim, flow_dim),
        )

        # 2. Shared Multimodal Latent Space Cross-Attention (Novel Component)
        self.multimodal_attn = SharedMultimodalCrossAttention(
            query_dim=flow_dim,
            key_val_dim=s2st_dim,
            out_dim=flow_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Final projection to velocity field condition Z_shared
        self.cond_proj = nn.Sequential(
            nn.Linear(flow_dim, flow_dim),
            nn.LayerNorm(flow_dim),
            nn.GELU(),
            nn.Linear(flow_dim, flow_dim),
        )

    def compute_video_envelope(self, source_frames: torch.Tensor) -> torch.Tensor:
        """
        Computes motion energy envelope from source video frames.
        
        Args:
            source_frames: (Batch, T_src, C, H, W) raw RGB video frames

        Returns:
            e_video: (Batch, T_src, video_dim) motion energy envelope
        """
        B, T, C, H, W = source_frames.shape

        # Compute temporal frame difference (motion velocity proxy)
        if T > 1:
            diff = torch.abs(source_frames[:, 1:] - source_frames[:, :-1]) # (B, T-1, C, H, W)
            # Pad last frame to keep T_src dimension
            diff = torch.cat([diff, diff[:, -1:]], dim=1)
        else:
            diff = source_frames

        # Spatial average pooling to get frame motion energy vector
        frame_diff_energy = diff.mean(dim=[-2, -1]) # (B, T, C)

        # 1D conv encoding along temporal dimension
        x = frame_diff_energy.transpose(1, 2)       # (B, C, T)
        e_video = self.video_envelope_encoder(x).transpose(1, 2) # (B, T_proj, video_dim)

        return e_video

    def forward(
        self,
        s2st_features: torch.Tensor,       # (Batch, T_speech, s2st_dim)
        source_frames: torch.Tensor,       # (Batch, T_video, C, H, W)
        speech_rate_alpha: float = 1.0,
    ) -> dict:
        """
        Forward pass of IsoSyncBridge.

        Args:
            s2st_features: Feature map from pretrained streaming S2ST speech encoder
            source_frames: Source video frames for extracting visual motion envelope
            speech_rate_alpha: Isochronous duration multiplier

        Returns:
            dict containing:
                - z_shared: (Batch, T_target, flow_dim) continuous multimodal latent condition
                - predicted_durations: (Batch, T_speech) predicted acoustic frame durations
                - log_durations: (Batch, T_speech) continuous log duration for loss
                - expanded_features: (Batch, T_target, s2st_dim) length-regulated speech features
        """
        B, T_speech, _ = s2st_features.shape

        # Step 1: Compute visual motion envelope from source video
        e_video = self.compute_video_envelope(source_frames) # (B, T_proj, video_dim)

        # Step 2: Isochronous Duration Prediction
        predicted_durations, log_durations = self.duration_predictor(
            h_speech=s2st_features,
            e_video=e_video,
            speech_rate_alpha=speech_rate_alpha,
        )

        # Step 3: Expand features according to predicted duration (Length Regulator)
        expanded_features = self.duration_predictor.expand_features(
            h_speech=s2st_features,
            durations=predicted_durations,
        ) # (B, T_target, s2st_dim)

        # Step 4: Generate visual queries aligned with target sequence length T_target
        T_target = expanded_features.shape[1]
        
        # Resample video envelope to match T_target
        e_video_resampled = F.interpolate(
            e_video.transpose(1, 2),
            size=T_target,
            mode="linear",
            align_corners=False
        ).transpose(1, 2) # (B, T_target, video_dim)

        visual_queries = self.visual_query_gen(e_video_resampled) # (B, T_target, flow_dim)

        # Step 5: Shared Multimodal Cross-Attention (Visual queries attend to expanded speech)
        multimodal_latent = self.multimodal_attn(
            visual_queries=visual_queries,
            speech_semantics=expanded_features,
        ) # (B, T_target, flow_dim)

        # Step 6: Velocity guidance condition Z_shared
        z_shared = self.cond_proj(multimodal_latent) # (B, T_target, flow_dim)

        return {
            "z_shared": z_shared,
            "predicted_durations": predicted_durations,
            "log_durations": log_durations,
            "expanded_features": expanded_features,
            "e_video": e_video,
        }
