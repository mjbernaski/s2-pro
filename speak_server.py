"""
Fish Audio S2 Pro - Server-side TTS worker

Loads model once, reads text lines from stdin, writes WAV bytes to stdout.
Protocol: for each input line, outputs "SIZE\n<wav_bytes>" or "ERROR: msg\n".
Prints "READY" when model is loaded.

Not meant to be run directly — used by speak_remote.py.
"""

import io
import os
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).parent.resolve()
FISH_SPEECH_DIR = MODEL_DIR.parent / "fish-speech"
sys.path.insert(0, str(FISH_SPEECH_DIR))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Redirect loguru to stderr so it doesn't corrupt the binary protocol on stdout
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def main():
    import torch
    import soundfile as sf

    from fish_speech.models.text2semantic.inference import (
        init_model,
        load_codec_model,
        generate_long,
        decode_to_audio,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.bfloat16

    log("Loading TTS model...")
    # Print loading status to stdout for the client to display
    sys.stdout.write("Loading TTS model...\n")
    sys.stdout.flush()

    model, decode_one_token = init_model(
        checkpoint_path=str(MODEL_DIR),
        device=device,
        precision=precision,
        compile=False,
    )

    sys.stdout.write("Loading codec...\n")
    sys.stdout.flush()
    codec = load_codec_model(str(MODEL_DIR / "codec.pth"), "cpu", torch.float32)

    log("Models loaded, signaling READY")
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    # Use binary mode for stdout to send wav bytes
    stdout_bin = sys.stdout.buffer

    # Redirect sys.stdout to stderr so any prints from generate_long
    # (e.g. visualization, tqdm) don't corrupt the binary protocol
    sys.stdout = sys.stderr

    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break

        text = line.strip()
        if not text:
            continue

        log(f"Generating: {text}")
        t0 = time.time()

        try:
            if not text.startswith("<|speaker:"):
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
                stdout_bin.write(b"ERROR: no audio generated\n")
                stdout_bin.flush()
                continue

            codes = torch.cat(all_codes, dim=1)
            audio = decode_to_audio(codes.to("cpu"), codec)

            # Encode to WAV in memory
            buf = io.BytesIO()
            sf.write(buf, audio.float().numpy(), codec.sample_rate, format="WAV")
            wav_bytes = buf.getvalue()

            elapsed = time.time() - t0
            log(f"Generated {len(wav_bytes)} bytes in {elapsed:.1f}s")

            # Send size then bytes
            stdout_bin.write(f"{len(wav_bytes)}\n".encode())
            stdout_bin.write(wav_bytes)
            stdout_bin.flush()

        except Exception as e:
            log(f"Error: {e}")
            stdout_bin.write(f"ERROR: {e}\n".encode())
            stdout_bin.flush()


if __name__ == "__main__":
    main()
