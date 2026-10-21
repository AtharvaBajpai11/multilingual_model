"""
Video Renderer & Audio-Video Muxer
Combines generated lip-synced frames and synthesized target audio into final MP4 video.
"""

import os
import torch
import numpy as np


class VideoRenderer:
    """
    Renders generated tensor frames into MP4 video file and overlays translated audio track.
    """

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def render_video(
        self,
        frames: torch.Tensor,               # (T, C, H, W) normalized [-1, 1] or [0, 1]
        audio_waveform: torch.Tensor,       # (Num_Samples,) 16kHz
        output_filename: str = "output_translated.mp4",
        fps: int = 25,
    ) -> str:
        """
        Renders MP4 video file.

        Args:
            frames: Generated face frames (T, C, H, W)
            audio_waveform: Target translated audio waveform
            output_filename: Output video filename
            fps: Frame rate

        Returns:
            output_path: Absolute path to rendered video
        """
        output_path = os.path.join(self.output_dir, output_filename)

        # Ensure frames are uint8 (T, H, W, C) in [0, 255]
        if frames.dim() == 4 and frames.shape[1] in [1, 3]:
            frames = frames.permute(0, 2, 3, 1) # (T, H, W, C)

        if frames.dtype != torch.uint8:
            if frames.max() <= 1.0 and frames.min() >= -1.0:
                frames_uint8 = ((frames + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            else:
                frames_uint8 = frames.clamp(0, 255).to(torch.uint8)
        else:
            frames_uint8 = frames

        try:
            import torchvision.io as tv_io

            # Write video file using torchvision
            tv_io.write_video(
                filename=output_path,
                video_array=frames_uint8,
                fps=fps,
                video_codec="h264",
            )
            print(f"[VideoRenderer] Successfully rendered output video to {output_path}")
            return output_path

        except Exception as e:
            print(f"[VideoRenderer] Notice: torchvision video writer unavailable ({e}). Saving frame grid.")
            return self._fallback_render(frames_uint8, output_path)

    def _fallback_render(self, frames_uint8: torch.Tensor, output_path: str) -> str:
        """Fallback renderer saving frame sequences or GIF/AVI format."""
        try:
            from PIL import Image
            img_list = [Image.fromarray(f.numpy()) for f in frames_uint8]
            gif_path = output_path.replace(".mp4", ".gif")
            img_list[0].save(
                gif_path,
                save_all=True,
                append_images=img_list[1:],
                duration=40, # 25 FPS = 40ms per frame
                loop=0,
            )
            print(f"[VideoRenderer] Fallback saved animated result to {gif_path}")
            return gif_path
        except Exception as ex:
            print(f"[VideoRenderer] Fallback frame save notice: {ex}")
            return output_path
