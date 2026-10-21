"""
IsoFlowS2ST Architecture
End-to-End Streaming Speech-to-Speech Translation & Flow-Matching Lip-Sync Model.
Integrates S2ST Backbone Encoder + IsoSyncBridge + DiFlowDubber Lip-Sync Decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.s2st_encoder import StreamingS2STEncoder
from .backbones.diflow_dubber import DiFlowDubberDecoder
from .bridge.isosync_bridge import IsoSyncBridge


class IsoFlowS2ST(nn.Module):
    """
    IsoFlow-S2ST Full Model Architecture.
    
    Combines:
    - Pretrained Streaming S2ST Speech Encoder (Frozen / Lightly Adapted)
    - Novel IsoSync-Bridge (Trainable Connecting Module)
    - Pretrained Flow-Matching Lip-Sync Decoder (Frozen / Lightly Adapted)
    """

    def __init__(
        self,
        audio_input_dim: int = 80,
        s2st_dim: int = 512,
        flow_dim: int = 512,
        video_dim: int = 64,
        freeze_backbones: bool = True,
    ):
        super().__init__()
        self.freeze_backbones = freeze_backbones

        # 1. Pretrained Streaming S2ST Speech Encoder Backbone
        self.s2st_encoder = StreamingS2STEncoder(
            input_dim=audio_input_dim,
            hidden_dim=s2st_dim,
            num_layers=6,
            num_heads=8,
            streaming=True,
        )

        # 2. NOVEL CONNECTING MODULE: IsoSync-Bridge
        self.isosync_bridge = IsoSyncBridge(
            s2st_dim=s2st_dim,
            flow_dim=flow_dim,
            video_dim=video_dim,
            num_heads=8,
        )

        # 3. Pretrained Flow-Matching Lip-Sync Decoder Backbone
        self.diflow_decoder = DiFlowDubberDecoder(
            frame_channels=3,
            hidden_dim=flow_dim,
            cond_dim=flow_dim,
            num_blocks=4,
            spatial_size=64,
        )

        # Apply freezing strategy
        if freeze_backbones:
            self.freeze_backbones_strategy()

    def freeze_backbones_strategy(self):
        """Freezes S2ST Encoder & DiFlow Decoder parameters, leaving ONLY IsoSyncBridge trainable."""
        for param in self.s2st_encoder.parameters():
            param.requires_grad = False

        for param in self.diflow_decoder.parameters():
            param.requires_grad = False

        # IsoSync-Bridge remains fully trainable
        for param in self.isosync_bridge.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        """Returns generator of trainable parameters for optimizer."""
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward(
        self,
        audio_features: torch.Tensor,     # (Batch, Mel_Bins, T_speech)
        source_frames: torch.Tensor,      # (Batch, T_video, C, H, W)
        ref_face: torch.Tensor,           # (Batch, C, H, W) reference face frame
        x_t: torch.Tensor = None,         # Noisy target face frames for flow training (Batch, T_target, C, H, W)
        t: torch.Tensor = None,           # Timestep tensor (Batch, T_target)
        speech_rate_alpha: float = 1.0,
    ) -> dict:
        """
        Forward pass of IsoFlow-S2ST.

        Returns:
            dict containing:
                - v_t: Predicted velocity field from DiFlowDubber (if x_t & t provided)
                - z_shared: Multimodal latent representation
                - predicted_durations: Predicted frame durations
                - log_durations: Continuous log durations for loss calculation
        """
        # Step 1: S2ST Encoder extracts streaming semantic acoustic representation
        if self.freeze_backbones:
            with torch.no_grad():
                h_s2st = self.s2st_encoder(audio_features)
        else:
            h_s2st = self.s2st_encoder(audio_features)

        # Step 2: IsoSync-Bridge computes Z_shared and duration control
        bridge_out = self.isosync_bridge(
            s2st_features=h_s2st,
            source_frames=source_frames,
            speech_rate_alpha=speech_rate_alpha,
        )

        z_shared = bridge_out["z_shared"] # (Batch, T_target, flow_dim)

        v_t = None
        if x_t is not None and t is not None:
            B, T_tgt, C, H, W = x_t.shape

            # Resample z_shared along time dimension if T_target != T_tgt
            if z_shared.shape[1] != T_tgt:
                z_shared = F.interpolate(
                    z_shared.transpose(1, 2),
                    size=T_tgt,
                    mode="linear",
                    align_corners=False
                ).transpose(1, 2)

            # Flatten Batch and Time dimensions for 2D Conv flow matching decoder
            x_t_flat = x_t.reshape(B * T_tgt, C, H, W)
            t_flat = t.reshape(B * T_tgt)
            
            # Repeat reference face across target time dimension
            ref_face_flat = ref_face.unsqueeze(1).repeat(1, T_tgt, 1, 1, 1).reshape(B * T_tgt, C, H, W)
            z_shared_flat = z_shared.reshape(B * T_tgt, -1)

            # Step 3: DiFlow Decoder predicts conditional velocity field v_t
            if self.freeze_backbones:
                v_t_flat = self.diflow_decoder(x_t_flat, t_flat, ref_face_flat, z_shared_flat)
            else:
                v_t_flat = self.diflow_decoder(x_t_flat, t_flat, ref_face_flat, z_shared_flat)

            v_t = v_t_flat.reshape(B, T_tgt, C, H, W)

        return {
            "v_t": v_t,
            "z_shared": z_shared,
            "predicted_durations": bridge_out["predicted_durations"],
            "log_durations": bridge_out["log_durations"],
            "expanded_features": bridge_out["expanded_features"],
        }

    @torch.no_grad()
    def generate_lip_sync(
        self,
        audio_features: torch.Tensor,
        source_frames: torch.Tensor,
        ref_face: torch.Tensor,
        num_flow_steps: int = 10,
        speech_rate_alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Inference routine solving OT-Flow ODE to generate translated lip-synced video frames.

        Returns:
            generated_video_frames: (Batch, T_target, C, H, W)
        """
        # Step 1: Speech Encoder
        h_s2st = self.s2st_encoder(audio_features)

        # Step 2: IsoSync-Bridge
        bridge_out = self.isosync_bridge(
            s2st_features=h_s2st,
            source_frames=source_frames,
            speech_rate_alpha=speech_rate_alpha,
        )

        z_shared = bridge_out["z_shared"] # (B, T_target, flow_dim)
        B, T_tgt, flow_dim = z_shared.shape
        _, C, H, W = ref_face.shape

        # Flatten for decoder ODE sampling
        ref_face_flat = ref_face.unsqueeze(1).repeat(1, T_tgt, 1, 1, 1).reshape(B * T_tgt, C, H, W)
        z_shared_flat = z_shared.reshape(B * T_tgt, flow_dim)

        # Step 3: Solve Flow-Matching ODE
        generated_flat = self.diflow_decoder.sample_ode(
            ref_face=ref_face_flat,
            z_shared=z_shared_flat,
            num_steps=num_flow_steps,
        )

        generated_video_frames = generated_flat.reshape(B, T_tgt, C, H, W)
        return generated_video_frames
