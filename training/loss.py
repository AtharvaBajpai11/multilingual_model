"""
IsoFlowS2ST Composite Loss Function
Combines Optimal Transport Flow Matching Loss (L_FM), Isochronous Duration Loss (L_dur),
and Cross-Modal Latent Alignment Loss (L_lat).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IsoFlowS2STLoss(nn.Module):
    """
    Composite Loss for IsoFlow-S2ST Fine-Tuning.

    Loss Formulation:
    L_total = lambda_fm * L_FM + lambda_dur * L_dur + lambda_lat * L_lat
    """

    def __init__(
        self,
        lambda_fm: float = 1.0,     # Weight for Flow Matching Velocity MSE
        lambda_dur: float = 0.5,    # Weight for Isochronous Duration Loss
        lambda_lat: float = 0.1,    # Weight for Latent Cosine Alignment
    ):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_dur = lambda_dur
        self.lambda_lat = lambda_lat

    def compute_flow_matching_target(
        self,
        x_0: torch.Tensor,   # Noise frame x_0 ~ N(0, I)
        x_1: torch.Tensor,   # Ground truth target video frame x_1
        t: torch.Tensor,     # Continuous timestep in [0, 1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optimal Transport linear interpolation trajectory:
        x_t = (1 - t) * x_0 + t * x_1
        Target velocity vector field:
        u_t = x_1 - x_0
        """
        # Ensure timestep matching shape (Batch, T, 1, 1, 1)
        t_expand = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        u_t = x_1 - x_0
        return x_t, u_t

    def forward(
        self,
        model_outputs: dict,
        x_0: torch.Tensor,               # Initial noise x_0 (Batch, T_tgt, C, H, W)
        x_1: torch.Tensor,               # Ground truth target frames x_1 (Batch, T_tgt, C, H, W)
        target_durations: torch.Tensor,  # Ground truth acoustic durations (Batch, T_speech)
        u_t: torch.Tensor,               # Target velocity field x_1 - x_0
    ) -> dict:
        """
        Computes composite loss dictionary.

        Args:
            model_outputs: Dict returned by IsoFlowS2ST forward pass
            x_0: Noise tensor
            x_1: Ground truth target frames
            target_durations: Target frame counts
            u_t: Target flow vector field

        Returns:
            dict containing total_loss, l_fm, l_dur, and l_lat
        """
        v_t = model_outputs["v_t"]                     # (Batch, T_tgt, C, H, W)
        log_durations = model_outputs["log_durations"] # (Batch, T_speech)
        z_shared = model_outputs["z_shared"]           # (Batch, T_tgt, flow_dim)

        # 1. Flow Matching Loss: MSE(v_t, u_t)
        if v_t is not None:
            l_fm = F.mse_loss(v_t, u_t)
        else:
            l_fm = torch.tensor(0.0, device=log_durations.device)

        # 2. Isochronous Duration Loss: Smooth L1 / MSE loss on log durations
        log_target_durations = torch.log(torch.clamp(target_durations.float(), min=1.0))
        
        # Align sequence lengths if needed
        if log_durations.shape[1] != log_target_durations.shape[1]:
            min_len = min(log_durations.shape[1], log_target_durations.shape[1])
            log_durations = log_durations[:, :min_len]
            log_target_durations = log_target_durations[:, :min_len]

        l_dur = F.smooth_l1_loss(log_durations, log_target_durations)

        # 3. Latent Alignment Loss: Variance regularization on Z_shared to prevent mode collapse
        latent_var = torch.var(z_shared, dim=-1)
        l_lat = torch.mean(F.relu(1.0 - latent_var))

        # Total weighted loss
        total_loss = (
            self.lambda_fm * l_fm +
            self.lambda_dur * l_dur +
            self.lambda_lat * l_lat
        )

        return {
            "loss": total_loss,
            "l_fm": l_fm,
            "l_dur": l_dur,
            "l_lat": l_lat,
        }
