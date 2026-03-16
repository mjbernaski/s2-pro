"""
Fish Audio S2 Pro - Interactive TTS CLI

Type text, hear it spoken. Ctrl+C to quit.
"""

import os
import sys
import tempfile
import subprocess
import time
from pathlib import Path

MODEL_DIR = Path(__file__).parent.resolve()
FISH_SPEECH_DIR = MODEL_DIR.parent / "fish-speech"
sys.path.insert(0, str(FISH_SPEECH_DIR))

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_models(device, precision):
    import torch
    from fish_speech.models.text2semantic.inference import (
        init_model,
        load_codec_model,
    )

    print("Loading TTS model...")
    t0 = time.time()
    model, decode_one_token = init_model(
        checkpoint_path=str(MODEL_DIR),
        device=device,
        precision=precision,
        compile=False,
    )
    print(f"  TTS model loaded in {time.time() - t0:.1f}s")

    print("Loading audio codec...")
    t0 = time.time()
    codec = load_codec_model(str(MODEL_DIR / "codec.pth"), device, precision)
    print(f"  Codec loaded in {time.time() - t0:.1f}s")

    return model, decode_one_token, codec


def speak(text, model, decode_one_token, codec, device):
    import torch
    from fish_speech.models.text2semantic.inference import (
        generate_long,
        decode_to_audio,
    )

    # Ensure speaker tag
    if not text.strip().startswith("<|speaker:"):
        text = f"<|speaker:0|>{text}"

    all_codes = []
    for response in generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=0,
        top_p=0.9,
        top_k=30,
        temperature=0.7,
        compile=False,
        iterative_prompt=True,
        chunk_length=300,
        prompt_text=None,
        prompt_tokens=None,
    ):
        if response.action == "sample":
            all_codes.append(response.codes)
        elif response.action == "next":
            break

    if not all_codes:
        print("  (no audio generated)")
        return

    codes = torch.cat(all_codes, dim=1)
    audio = decode_to_audio(codes.to(device), codec)

    # Write to temp file and play
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        import soundfile as sf
        sf.write(tmp_path, audio.cpu().float().numpy(), codec.sample_rate)

    duration = audio.shape[0] / codec.sample_rate
    print(f"  ({duration:.1f}s audio)")
    subprocess.run(["afplay", tmp_path])
    os.unlink(tmp_path)


def main():
    import torch

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    precision = torch.bfloat16
    model, decode_one_token, codec = load_models(device, precision)

    print()
    print("Ready! Type text to speak. Ctrl+C to quit.")
    print("Tip: use [tags] for expression, e.g. [whisper]hello [excited]wow!")
    print()

    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                continue

            t0 = time.time()
            speak(text, model, decode_one_token, codec, device)
            elapsed = time.time() - t0
            print(f"  (total: {elapsed:.1f}s)\n")
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
