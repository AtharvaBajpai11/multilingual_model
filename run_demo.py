"""
IsoFlow-S2ST End-to-End Demo Script
Demonstrates fine-tuning the novel IsoSyncBridge connecting module with frozen backbones,
followed by multi-language inference and lip-sync video rendering.
"""

import torch
from torch.utils.data import DataLoader

from models.isoflow_s2st import IsoFlowS2ST
from data.dataset import SyntheticMultilingualVideoDataset
from training.trainer import IsoFlowTrainer
from pipeline.translator import MultilingualVideoTranslator
from pipeline.video_processor import VideoProcessor
from pipeline.video_renderer import VideoRenderer


def main():
    print("=" * 70)
    print(" IsoFlow-S2ST Architecture & Fine-Tuning Demo ")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    # 1. Instantiate Model & Freeze Strategy
    print("\n[Step 1] Instantiating IsoFlow-S2ST Model...")
    model = IsoFlowS2ST(
        audio_input_dim=80,
        s2st_dim=512,
        flow_dim=512,
        freeze_backbones=True,
    ).to(device)

    # Print Parameter Counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    frozen_params = total_params - trainable_params

    print(f"  - Total Parameters:      {total_params:,}")
    print(f"  - Frozen Backbone Params: {frozen_params:,} (S2ST Speech Encoder + DiFlow Decoder)")
    print(f"  - Trainable Bridge Params: {trainable_params:,} (IsoSyncBridge)")

    # 2. Dataset & Dataloader
    print("\n[Step 2] Initializing Synthetic Multimodal Audio-Video Dataset...")
    dataset = SyntheticMultilingualVideoDataset(num_samples=16)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 3. Fine-Tune IsoSyncBridge
    print("\n[Step 3] Fine-tuning IsoSyncBridge connecting module (1 Epoch)...")
    trainer = IsoFlowTrainer(
        model=model,
        train_dataloader=dataloader,
        lr=1e-4,
        device=device,
    )
    metrics = trainer.train_epoch(epoch=1)
    print(f"  - Training Loss: {metrics['loss']:.4f} (FM: {metrics['l_fm']:.4f}, Dur: {metrics['l_dur']:.4f})")
    trainer.save_checkpoint("isosync_bridge_demo.pt")

    # 4. Multi-Language Inference Test
    print("\n[Step 4] Running Multi-Language Translation & Lip-Sync Inference...")
    processor = VideoProcessor()
    inputs = processor._generate_dummy_inputs(num_frames=25)

    translator = MultilingualVideoTranslator(model=model, device=device)
    
    for lang in ["es", "fr", "hi"]:
        result = translator.translate_and_sync(inputs, target_lang=lang, flow_steps=5)
        print(f"  - Translated to {result['target_lang']}: Generated {result['generated_frames'].shape[0]} lip-synced frames")

    # 5. Render Sample Output Video
    print("\n[Step 5] Rendering Demo Translated Video...")
    renderer = VideoRenderer(output_dir="./output")
    out_path = renderer.render_video(
        frames=result["generated_frames"],
        audio_waveform=result["target_audio_waveform"],
        output_filename="demo_isoflow_output.mp4",
    )

    print("\n" + "=" * 70)
    print(f" DEMO COMPLETE: Output saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
