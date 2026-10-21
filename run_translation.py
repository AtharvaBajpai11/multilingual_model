"""
CLI Command Line Interface for Multilingual Video Translation & Lip-Sync
Usage:
    python run_translation.py --input input.mp4 --target_lang es --output output_es.mp4
"""

import argparse
import sys
import os

from pipeline.video_processor import VideoProcessor
from pipeline.translator import MultilingualVideoTranslator, SUPPORTED_LANGUAGES
from pipeline.video_renderer import VideoRenderer


def main():
    parser = argparse.ArgumentParser(description="IsoFlow-S2ST Multilingual Video Translation & Lip-Sync CLI")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input pre-downloaded video (.mp4)")
    parser.add_argument("--target_lang", "-l", type=str, default="es", choices=list(SUPPORTED_LANGUAGES.keys()), help="Target language code (es, fr, de, hi, ja, zh, en)")
    parser.add_argument("--output", "-o", type=str, default="output_translated.mp4", help="Path to save output video")
    parser.add_argument("--steps", "-s", type=int, default=10, help="Flow Matching ODE solver steps")

    args = parser.parse_args()

    print("=" * 70)
    print(" IsoFlow-S2ST: Multilingual AI Video Translation & Isochronous Lip-Sync ")
    print("=" * 70)
    print(f"Input Video:        {args.input}")
    print(f"Target Language:    {SUPPORTED_LANGUAGES[args.target_lang]['name']} ({args.target_lang})")
    print(f"Flow Matching ODE:  {args.steps} integration steps")
    print(f"Connecting Module:  IsoSync-Bridge (Isochronous Multimodal Latent Space)")
    print("-" * 70)

    # Initialize components
    processor = VideoProcessor()
    translator = MultilingualVideoTranslator()
    renderer = VideoRenderer(output_dir=os.path.dirname(os.path.abspath(args.output)) or "./")

    # 1. Demux video into frames & audio spectrograms
    print("[1/3] Demuxing audio spectrograms & facial frame sequences...")
    processed_inputs = processor.extract_audio_and_frames(args.input)

    # 2. IsoFlow-S2ST Translation & Lip-Sync Generation
    print(f"[2/3] Executing IsoFlow-S2ST engine (Translating to {SUPPORTED_LANGUAGES[args.target_lang]['name']})...")
    result = translator.translate_and_sync(
        processed_inputs=processed_inputs,
        target_lang=args.target_lang,
        flow_steps=args.steps,
    )

    # 3. Render output video
    print("[3/3] Muxing generated lip-synced face frames with target translated audio...")
    output_path = renderer.render_video(
        frames=result["generated_frames"],
        audio_waveform=result["target_audio_waveform"],
        output_filename=os.path.basename(args.output),
    )

    print("=" * 70)
    print(f" SUCCESS: Rendered translated video saved to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
