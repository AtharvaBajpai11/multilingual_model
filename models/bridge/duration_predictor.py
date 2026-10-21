"""
Isochronous Duration Predictor Module
Predicts token-level and frame-level durations for target TTS output and lip-sync generation
guided by the source video's temporal motion envelope and acoustic energy profile.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IsochronousDurationPredictor(nn.Module):
    """
    Lightweight Duration & Pacing Controller.
    Jointly conditions target sequence length on:
    1. Speech Encoder representations (h_speech)
    2. Source video facial motion energy envelope (e_video)
    3. Global speech rate control parameter (alpha)
    """

    def __init__(
        self,
        speech_dim: int = 512,
        video_dim: int = 64,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.speech_dim = speech_dim
        self.video_dim = video_dim

        # Fusion layer for speech features and video motion features
        self.fusion = nn.Linear(speech_dim + video_dim, hidden_dim)

        # 1D Convolutional Duration Predictor Stack
        self.conv_stack = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Scalar Log-Duration projection head
        self.duration_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        h_speech: torch.Tensor,       # (Batch, T_speech, speech_dim)
        e_video: torch.Tensor,        # (Batch, T_speech, video_dim) video motion energy/envelope
        speech_rate_alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass predicting log durations per acoustic frame.

        Args:
            h_speech: Acoustic/semantic representation from S2ST backbone
            e_video: Video motion/envelope representation resampled to T_speech
            speech_rate_alpha: Pacing multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)

        Returns:
            predicted_durations: (Batch, T_speech) predicted integer frame durations
            log_durations: (Batch, T_speech) continuous log duration for loss calculation
        """
        # Align video envelope length if necessary
        if e_video.shape[1] != h_speech.shape[1]:
            e_video = F.interpolate(
                e_video.transpose(1, 2),
                size=h_speech.shape[1],
                mode="linear",
                align_corners=False
            ).transpose(1, 2)

        # Concatenate acoustic & visual temporal envelope representations
        combined = torch.cat([h_speech, e_video], dim=-1) # (Batch, T_speech, speech_dim + video_dim)
        x = self.fusion(combined)                         # (Batch, T_speech, hidden_dim)

        # Apply 1D conv stack along time dimension
        x_conv = x.transpose(1, 2)                         # (Batch, hidden_dim, T_speech)
        x_conv = self.conv_stack(x_conv).transpose(1, 2)   # (Batch, T_speech, hidden_dim)

        # Predict continuous log duration
        log_durations = self.duration_head(x_conv).squeeze(-1) # (Batch, T_speech)

        # Convert to positive integer frame durations scaled by pacing alpha
        predicted_durations = torch.clamp(
            torch.round(torch.exp(log_durations) * speech_rate_alpha),
            min=1.0
        )

        return predicted_durations, log_durations

    def expand_features(
        self,
        h_speech: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expands acoustic features according to predicted frame durations (Length Regulator).
        """
        batch_size, seq_len, dim = h_speech.shape
        expanded_list = []

        for b in range(batch_size):
            h_b = h_speech[b]
            dur_b = durations[b].long()
            
            # Repeat each token according to its duration
            expanded_b = torch.repeat_interleave(h_b, dur_b, dim=0)
            expanded_list.append(expanded_b)

        # Pad batch to max expanded length
        max_len = max([exp.shape[0] for exp in expanded_list])
        expanded_padded = torch.zeros(batch_size, max_len, dim, device=h_speech.device)

        for b in range(batch_size):
            cur_len = expanded_list[b].shape[0]
            expanded_padded[b, :cur_len] = expanded_list[b]

        return expanded_padded
