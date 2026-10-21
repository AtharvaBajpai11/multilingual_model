"""
IsoFlowS2ST Trainer Module
Handles fine-tuning of the IsoSync-Bridge connecting module with frozen backbones.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from .loss import IsoFlowS2STLoss


class IsoFlowTrainer:
    """
    Fine-Tuning Trainer for IsoFlow-S2ST.
    Only updates trainable parameters in IsoSync-Bridge while keeping S2ST & DiFlow backbones frozen.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Loss function
        self.criterion = IsoFlowS2STLoss()

        # Optimizer targeting ONLY trainable parameters (IsoSyncBridge)
        trainable_params = list(model.get_trainable_parameters())
        print(f"[IsoFlowTrainer] Total trainable parameters in IsoSyncBridge: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-6,
        )

    def train_epoch(self, epoch: int) -> dict:
        """Trains for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_fm = 0.0
        total_dur = 0.0
        total_lat = 0.0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            audio_features = batch["audio_features"].to(self.device)
            source_frames = batch["source_frames"].to(self.device)
            ref_face = batch["ref_face"].to(self.device)
            target_frames = batch["target_frames"].to(self.device) # Ground truth x_1
            target_durations = batch["target_durations"].to(self.device)

            B, T_tgt, C, H, W = target_frames.shape

            # Sample random timesteps t ~ Uniform(0, 1)
            t = torch.rand(B, T_tgt, device=self.device)

            # Sample noise x_0 ~ N(0, I)
            x_0 = torch.randn_like(target_frames)

            # Compute flow target x_t and target velocity field u_t = x_1 - x_0
            x_t, u_t = self.criterion.compute_flow_matching_target(x_0, target_frames, t)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                audio_features=audio_features,
                source_frames=source_frames,
                ref_face=ref_face,
                x_t=x_t,
                t=t,
            )

            # Compute loss
            loss_dict = self.criterion(
                model_outputs=outputs,
                x_0=x_0,
                x_1=target_frames,
                target_durations=target_durations,
                u_t=u_t,
            )

            loss = loss_dict["loss"]
            loss.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_fm += loss_dict["l_fm"].item()
            total_dur += loss_dict["l_dur"].item()
            total_lat += loss_dict["l_lat"].item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "l_fm": f"{loss_dict['l_fm'].item():.4f}",
                "l_dur": f"{loss_dict['l_dur'].item():.4f}",
            })

        self.scheduler.step()

        num_batches = len(self.train_dataloader)
        return {
            "loss": total_loss / num_batches,
            "l_fm": total_fm / num_batches,
            "l_dur": total_dur / num_batches,
            "l_lat": total_lat / num_batches,
        }

    def save_checkpoint(self, filename: str = "isosync_bridge_checkpoint.pt"):
        """Saves weights of the IsoSyncBridge connecting module."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "isosync_bridge_state_dict": self.model.isosync_bridge.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"[IsoFlowTrainer] Saved IsoSyncBridge checkpoint to {path}")
