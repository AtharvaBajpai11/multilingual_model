"""
Multimodal Dataset Pipeline & Synthetic Data Generator
Provides PyTorch Dataset implementations for training and testing IsoFlow-S2ST.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SyntheticMultilingualVideoDataset(Dataset):
    """
    Synthetic Multimodal Audio-Video Dataset for testing and prototyping IsoFlow-S2ST.
    Generates aligned dummy audio mel-spectrograms, source face frames, reference face images, 
    and ground truth target lip-synced face frames.
    """

    def __init__(
        self,
        num_samples: int = 32,
        t_speech: int = 50,      # Number of speech mel-spectrogram frames
        t_video: int = 25,       # Number of source video frames (1 sec @ 25 FPS)
        t_target: int = 25,      # Number of target lip-sync frames
        mel_bins: int = 80,      # Mel-spectrogram frequency channels
        spatial_size: int = 64,  # Face image crop size
    ):
        super().__init__()
        self.num_samples = num_samples
        self.t_speech = t_speech
        self.t_video = t_video
        self.t_target = t_target
        self.mel_bins = mel_bins
        self.spatial_size = spatial_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        # Generate random seed per index for reproducibility
        rng = np.random.RandomState(idx)

        # Synthetic Log-Mel Spectrogram: (Mel_Bins, T_speech)
        audio_features = torch.from_numpy(
            rng.randn(self.mel_bins, self.t_speech).astype(np.float32)
        )

        # Synthetic Source Video Face Frames: (T_video, 3, 64, 64)
        source_frames = torch.from_numpy(
            rng.randn(self.t_video, 3, self.spatial_size, self.spatial_size).astype(np.float32)
        )

        # Synthetic Reference Face Frame: (3, 64, 64)
        ref_face = torch.from_numpy(
            rng.randn(3, self.spatial_size, self.spatial_size).astype(np.float32)
        )

        # Synthetic Target Ground Truth Lip-Sync Face Frames: (T_target, 3, 64, 64)
        target_frames = torch.from_numpy(
            rng.randn(self.t_target, 3, self.spatial_size, self.spatial_size).astype(np.float32)
        )

        # Target acoustic frame durations (e.g. 1 frame duration per speech step)
        target_durations = torch.ones(self.t_speech, dtype=torch.float32)

        return {
            "audio_features": audio_features,
            "source_frames": source_frames,
            "ref_face": ref_face,
            "target_frames": target_frames,
            "target_durations": target_durations,
        }


class MultimodalVideoDataset(Dataset):
    """
    Real Video Dataset placeholder for real video directory loading.
    Extracts frames and audio spectrograms using ffmpeg/torchaudio.
    """

    def __init__(self, video_paths: list):
        super().__init__()
        self.video_paths = video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        # Implementation for real file reading
        raise NotImplementedError("Real file dataset reading is handled by pipeline.video_processor")
