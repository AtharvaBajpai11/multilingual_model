"""
IsoFlow-S2ST: User Audio Upload & English Translation Tool
Accepts any user audio file (.wav, .mp3, etc.) and generates a translated audio file in English.

Usage:
    python translate_audio.py --input path/to/your_audio.wav --output translated_english.wav
    python translate_audio.py   (Interactive mode - prompts for file or generates demo audio)
"""

import argparse
import os
import sys

from pipeline.audio_translator import AudioTranslator


def main():
    parser = argparse.ArgumentParser(description="Translate input speech audio into English using IsoFlow-S2ST")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input audio file (.wav, .mp3, etc.)")
    parser.add_argument("--output", "-o", type=str, default="translated_english.wav", help="Path to save translated English audio")
    parser.add_argument("--demo", action="store_true", help="Generate and test with synthetic demo audio")

    args = parser.parse_args()

    print("=" * 70)
    print("   IsoFlow-S2ST: Audio Speech Translation to English   ")
    print("=" * 70)

    input_file = args.input

    # If demo flag or input specified as demo_source_speech.wav and not exists
    if args.demo or (input_file == "demo_source_speech.wav" and not os.path.exists("demo_source_speech.wav")):
        print("Generating demo source speech audio ('demo_source_speech.wav')...")
        input_file = AudioTranslator.generate_demo_audio("demo_source_speech.wav", duration_sec=3.5)

    # Interactive input handling if no argument was passed
    elif input_file is None:
        print("\nNo input audio file specified via command line.")
        user_choice = input("Enter the path to your audio file (or press ENTER to test with demo audio): ").strip()

        if user_choice and os.path.exists(user_choice):
            input_file = user_choice
        else:
            if user_choice:
                print(f"File '{user_choice}' not found.")
            print("Generating demo source speech audio ('demo_source_speech.wav') for testing...")
            input_file = AudioTranslator.generate_demo_audio("demo_source_speech.wav", duration_sec=3.5)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    print(f"\n[1/3] Loading input audio: {input_file}")
    translator = AudioTranslator()

    # Perform translation to English
    print("[2/3] Translating speech to English using IsoFlow-S2ST backbone...")
    result = translator.translate_to_english(audio_path=input_file)

    # Save translated English audio
    print(f"[3/3] Saving translated English audio to: {args.output}")
    translator.save_audio(result["translated_waveform"], args.output)

    # Print summary metrics
    print("\n" + "=" * 70)
    print("   TRANSLATION COMPLETE   ")
    print("=" * 70)
    print(f"  - Source File:        {input_file}")
    print(f"  - Target Language:    {result['metrics']['target_language']}")
    print(f"  - Target Duration:    {result['metrics']['target_duration_sec']:.2f} seconds")
    print(f"  - Sample Rate:        {result['sample_rate']} Hz (Mono WAV)")
    print(f"  - Output Saved At:    {os.path.abspath(args.output)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
