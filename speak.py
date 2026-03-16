"""
Fish Audio S2 Pro - Interactive TTS CLI

Type text, hear it spoken. Ctrl+C to quit.

Modes:
  --api     Use Fish Audio API (fast, requires FISH_API_KEY env var)
  --local   Use local model on MPS/CUDA (slow, no API key needed)

Options:
  --out DIR   Save audio files to DIR (default: ./output)
  --no-save   Don't save files, just play

Default: --api if FISH_API_KEY is set, otherwise --local
Audio files saved as output/001.wav, output/002.wav, etc.
"""

import os
import sys
import tempfile
import subprocess
import time
from pathlib import Path

MODEL_DIR = Path(__file__).parent.resolve()
FISH_SPEECH_DIR = MODEL_DIR.parent / "fish-speech"


# ── API mode ──────────────────────────────────────────────

def save_and_play(audio_bytes, save_path=None):
    """Write audio bytes to save_path (if given) and play it."""
    if save_path:
        with open(save_path, "wb") as f:
            f.write(audio_bytes)
        print(f"  saved: {save_path}")
        subprocess.run(["afplay", str(save_path)])
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        subprocess.run(["afplay", tmp_path])
        os.unlink(tmp_path)


def speak_api(text, api_key, save_path=None):
    import urllib.request
    import json

    body = json.dumps({
        "text": text,
        "format": "wav",
        "model": "s2-pro",
    }).encode()

    req = urllib.request.Request(
        "https://api.fish.audio/v1/tts",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(req) as resp:
        if resp.status != 200:
            print(f"  API error: {resp.status} {resp.read().decode()}")
            return
        audio_bytes = resp.read()

    save_and_play(audio_bytes, save_path)


def next_save_path(out_dir):
    """Return the next numbered file path like output/001.wav."""
    if out_dir is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(Path(out_dir).glob("*.wav"))
    n = len(existing) + 1
    return Path(out_dir) / f"{n:03d}.wav"


def run_api_mode(api_key, out_dir):
    print("Mode: API (fast)")
    print("Ready! Type text to speak. Ctrl+C to quit.")
    print("Tip: use [tags] like [whisper], [excited], [laughing]\n")

    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                continue

            t0 = time.time()
            speak_api(text, api_key, save_path=next_save_path(out_dir))
            print(f"  ({time.time() - t0:.1f}s)\n")
    except KeyboardInterrupt:
        print("\nBye!")


# ── Local mode ────────────────────────────────────────────

def load_models(device, precision):
    sys.path.insert(0, str(FISH_SPEECH_DIR))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    # Use CPU for codec to avoid NVRTC/arch issues on newer GPUs
    codec = load_codec_model(str(MODEL_DIR / "codec.pth"), "cpu", torch.float32)
    print(f"  Codec loaded in {time.time() - t0:.1f}s")

    return model, decode_one_token, codec


def speak_local(text, model, decode_one_token, codec, device, save_path=None):
    import torch
    from fish_speech.models.text2semantic.inference import (
        generate_long,
        decode_to_audio,
    )

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
    audio = decode_to_audio(codes.to("cpu"), codec)

    import soundfile as sf
    import io
    buf = io.BytesIO()
    sf.write(buf, audio.cpu().float().numpy(), codec.sample_rate, format="WAV")
    audio_bytes = buf.getvalue()

    duration = audio.shape[0] / codec.sample_rate
    print(f"  ({duration:.1f}s audio)")
    save_and_play(audio_bytes, save_path)


def run_local_mode(out_dir):
    import torch

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Mode: local ({device})")
    print("Note: ~1 tok/s on MPS, expect ~2 min per sentence\n")

    model, decode_one_token, codec = load_models(device, torch.bfloat16)

    print("\nReady! Type text to speak. Ctrl+C to quit.")
    print("Tip: use [tags] like [whisper], [excited], [laughing]\n")

    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                continue

            t0 = time.time()
            speak_local(text, model, decode_one_token, codec, device,
                        save_path=next_save_path(out_dir))
            print(f"  (total: {time.time() - t0:.1f}s)\n")
    except KeyboardInterrupt:
        print("\nBye!")


# ── Main ──────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    force_api = "--api" in args
    force_local = "--local" in args
    no_save = "--no-save" in args
    api_key = os.environ.get("FISH_API_KEY", "")

    # Parse --out DIR
    out_dir = None if no_save else "output"
    if "--out" in args:
        idx = args.index("--out")
        if idx + 1 < len(args):
            out_dir = args[idx + 1]

    if force_local:
        use_api = False
    elif force_api:
        if not api_key:
            print("Error: --api requires FISH_API_KEY environment variable")
            print("  Get a key at https://fish.audio and run:")
            print("  export FISH_API_KEY=your_key_here")
            sys.exit(1)
        use_api = True
    else:
        use_api = bool(api_key)

    print("Fish Audio S2 Pro - Interactive TTS\n")
    if out_dir:
        print(f"Saving audio to: {out_dir}/")

    if use_api:
        run_api_mode(api_key, out_dir)
    else:
        if not api_key:
            print("Tip: set FISH_API_KEY for fast API mode (~2s vs ~2min)")
            print("  Get a key at https://fish.audio")
            print("  export FISH_API_KEY=your_key_here\n")
        run_local_mode(out_dir)


if __name__ == "__main__":
    main()
