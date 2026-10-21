"""
Interactive Audio Speech-to-Speech Translation to English
Allows the user to upload/input any foreign speech audio file, or pick from recommended languages
to synthesize and translate speech into English using the IsoFlow-S2ST architecture.

Usage:
    python run_audio_translator.py
"""

import os
import sys
import time

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pipeline.audio_translator import AudioTranslator


RECOMMENDED_LANGUAGES = [
    {
        "id": "1",
        "name": "Spanish (Espanol)",
        "code": "es",
        "why": "[TOP BENCHMARK] Highest translation accuracy. Syllable-timed cadence contrasts sharply with English stress-timing.",
        "sample_file": "sample_spanish.wav",
    },
    {
        "id": "2",
        "name": "French (Francais)",
        "code": "fr",
        "why": "[PHONETIC DIVERSITY] Tests smooth acoustic transitions, liaison, and vowel compression.",
        "sample_file": "sample_french.wav",
    },
    {
        "id": "3",
        "name": "Hindi",
        "code": "hi",
        "why": "[CROSS-FAMILY GENERALIZATION] Tests non-Latin phonetics, retroflex consonants, and rich harmonic structures.",
        "sample_file": "sample_hindi.wav",
    },
    {
        "id": "4",
        "name": "German (Deutsch)",
        "code": "de",
        "why": "[STRESS-TIMED PACING] Tests compound word duration handling and plosive consonant preservation.",
        "sample_file": "sample_german.wav",
    },
    {
        "id": "5",
        "name": "Japanese",
        "code": "ja",
        "why": "[MORA-TIMED RYTHM] High mora velocity tests the IsoSync duration bridge's ability to normalize syllable density.",
        "sample_file": "sample_japanese.wav",
    },
]


def print_banner():
    print("\n" + "=" * 76)
    print("      [IsoFlow-S2ST] Interactive Foreign Speech to English Translator  ")
    print("=" * 76)
    print("  This tool translates speech audio from any language directly into English.")
    print("  Powered by: Streaming S2ST Backbone + Trainable IsoSync-Bridge + Neural Synth\n")


def display_recommendations():
    print("-" * 76)
    print("  RECOMMENDED LANGUAGES FOR TESTING THE TRANSLATOR:")
    print("-" * 76)
    for lang in RECOMMENDED_LANGUAGES:
        print(f"  [{lang['id']}] {lang['name']}")
        print(f"      • Recommendation: {lang['why']}")
    print("-" * 76)


def get_user_audio_input() -> str:
    display_recommendations()

    print("\nHOW WOULD YOU LIKE TO TEST?")
    print("  • Option A: Enter/Drag & Drop your own audio file path (.wav, .mp3, .m4a)")
    print("  • Option B: Type [1, 2, 3, 4, or 5] to auto-generate a sample in that language")
    print("  • Option C: Press ENTER directly to test with recommended Spanish [1]")

    user_input = input("\nEnter your choice: ").strip()

    # Clean enclosing quotes (Windows Drag-and-Drop adds quotes)
    cleaned_input = user_input.strip('"\' ')

    # Check if user picked a number [1-5]
    for lang in RECOMMENDED_LANGUAGES:
        if cleaned_input == lang["id"] or cleaned_input.lower() == lang["code"]:
            print(f"\n[Selection] Generating sample speech in {lang['name']}...")
            sample_path = AudioTranslator.generate_demo_audio(
                output_path=lang["sample_file"],
                duration_sec=3.5,
                language=lang["code"],
            )
            return sample_path

    # If empty, default to Spanish (Option 1)
    if not cleaned_input:
        default_lang = RECOMMENDED_LANGUAGES[0]
        print(f"\n[Default] Generating recommended sample in {default_lang['name']}...")
        sample_path = AudioTranslator.generate_demo_audio(
            output_path=default_lang["sample_file"],
            duration_sec=3.5,
            language=default_lang["code"],
        )
        return sample_path

    # If user provided a file path
    if os.path.exists(cleaned_input):
        return cleaned_input
    else:
        print(f"\n[Notice] File '{cleaned_input}' was not found.")
        print("[Fallback] Generating recommended Spanish speech sample ('sample_spanish.wav')...")
        sample_path = AudioTranslator.generate_demo_audio(
            output_path="sample_spanish.wav",
            duration_sec=3.5,
            language="es",
        )
        return sample_path


def main():
    print_banner()

    # 1. Ask user for input audio file
    input_file = get_user_audio_input()

    output_file = "translated_english_output.wav"

    print("\n" + "=" * 76)
    print(f"  [Step 1/3] Loading Input Audio: {os.path.basename(input_file)}")
    print("=" * 76)

    start_time = time.time()
    translator = AudioTranslator()

    print("\n" + "=" * 76)
    print("  [Step 2/3] Translating to English via IsoFlow-S2ST...")
    print("=" * 76)
    result = translator.translate_to_english(audio_path=input_file)

    print("\n" + "=" * 76)
    print("  [Step 3/3] Synthesizing & Saving Translated English Audio...")
    print("=" * 76)
    translator.save_audio(result["translated_waveform"], output_file)

    elapsed = time.time() - start_time

    # 2. Display complete output results and metrics
    print("\n" + "=" * 76)
    print("                     TRANSLATION SUCCEEDED                    ")
    print("=" * 76)
    print(f"  - Source Audio File:      {os.path.abspath(input_file)}")
    print(f"  - Target Language:        {result['metrics']['target_language']}")
    print(f"  - Translated Audio File:  {os.path.abspath(output_file)}")
    print(f"  - Target Audio Duration:  {result['metrics']['target_duration_sec']:.2f} seconds")
    print(f"  - Target Sample Rate:     {result['sample_rate']} Hz (Mono 16-bit WAV)")
    print(f"  - Processing Latency:     {elapsed:.2f} seconds")
    print("=" * 76)
    print(f"\nYou can now play your translated English audio at:\n-> {os.path.abspath(output_file)}\n")


if __name__ == "__main__":
    main()
