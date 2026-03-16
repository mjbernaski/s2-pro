"""
Fish Audio S2 Pro - Local Inference Test Script

Uses the fish-speech library for proper model loading and generation.

Stages:
  1. Import & tokenizer verification
  2. Model loading (DualARTransformer)
  3. Text-to-semantic-token generation
  4. Codec decoding (semantic tokens -> WAV audio)

Usage:
  python test_inference.py           # run all stages
  python test_inference.py 1         # tokenizer only
  python test_inference.py 1 2 3     # skip codec decoding
"""

import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).parent.resolve()
FISH_SPEECH_DIR = MODEL_DIR.parent / "fish-speech"

# Ensure fish-speech is importable
sys.path.insert(0, str(FISH_SPEECH_DIR))


def stage1_tokenizer():
    print("=" * 60)
    print("STAGE 1: Tokenizer & Imports")
    print("=" * 60)

    from fish_speech.tokenizer import FishTokenizer

    tokenizer = FishTokenizer(str(MODEL_DIR))
    print(f"  FishTokenizer loaded from {MODEL_DIR}")

    # Verify key tokens
    from fish_speech.tokenizer import (
        IM_END_TOKEN,
        IM_START_TOKEN,
        MODALITY_VOICE_TOKEN,
        AUDIO_START_TOKEN,
    )

    for tok in [IM_START_TOKEN, IM_END_TOKEN, MODALITY_VOICE_TOKEN, AUDIO_START_TOKEN]:
        tid = tokenizer.get_token_id(tok)
        print(f"  {tok} -> {tid}")

    # Verify semantic tokens
    sem0 = tokenizer.get_token_id("<|semantic:0|>")
    sem99 = tokenizer.get_token_id("<|semantic:99|>")
    print(f"  <|semantic:0|> -> {sem0}")
    print(f"  <|semantic:99|> -> {sem99}")

    # Test text encoding
    text = "Hello, this is a test."
    ids = tokenizer.encode(text)
    print(f"  Encoded '{text}' -> {len(ids)} tokens")

    print("  PASSED\n")
    return tokenizer


def stage2_model(device="mps"):
    print("=" * 60)
    print("STAGE 2: Model Loading")
    print("=" * 60)

    import torch
    from fish_speech.models.text2semantic.inference import init_model

    precision = torch.bfloat16
    print(f"  Loading DualARTransformer from {MODEL_DIR}")
    print(f"  Device: {device}, Precision: {precision}")

    t0 = time.time()
    model, decode_one_token = init_model(
        checkpoint_path=str(MODEL_DIR),
        device=device,
        precision=precision,
        compile=False,
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e9:.2f}B")
    print(f"  Config: {model.config.n_layer} layers, dim={model.config.dim}")
    print(f"  Codebooks: {model.config.num_codebooks}, size={model.config.codebook_size}")
    print(f"  Max seq len: {model.config.max_seq_len}")
    print("  PASSED\n")
    return model, decode_one_token


def stage3_generate(model, decode_one_token, device="mps"):
    print("=" * 60)
    print("STAGE 3: Text-to-Semantic Generation")
    print("=" * 60)

    import torch
    from fish_speech.models.text2semantic.inference import generate_long

    text = "<|speaker:0|>Hello! This is a test of Fish Audio S2 Pro. [excited] It works!"

    print(f"  Input text: {text}")
    print(f"  Generating (this may take a while on MPS)...")

    t0 = time.time()
    all_codes = []

    for response in generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=0,  # 0 = use model max
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
            print(f"  Generated chunk: {response.codes.shape} codes")
        elif response.action == "next":
            break

    gen_time = time.time() - t0

    if all_codes:
        codes = torch.cat(all_codes, dim=1)
        print(f"  Total codes shape: {codes.shape}")
        print(f"  Generation time: {gen_time:.1f}s")
        print(f"  Speed: {codes.shape[1] / gen_time:.1f} frames/s")
        print(f"  ~Audio duration: {codes.shape[1] / 21:.1f}s (at ~21 Hz frame rate)")
        print("  PASSED\n")
        return codes
    else:
        print("  WARNING: No codes generated")
        print("  FAILED\n")
        return None


def stage4_codec(codes, device="mps"):
    print("=" * 60)
    print("STAGE 4: Codec Decoding -> Audio")
    print("=" * 60)

    import torch

    codec_path = MODEL_DIR / "codec.pth"
    if not codec_path.exists():
        print(f"  SKIPPED: {codec_path} not found")
        return

    if codes is None:
        print("  SKIPPED: No codes to decode")
        return

    print(f"  Loading codec from {codec_path}...")
    t0 = time.time()

    from fish_speech.models.text2semantic.inference import load_codec_model

    # Use CPU for codec — avoids NVRTC issues on newer GPUs
    codec_device = "cpu"
    precision = torch.float32 if codec_device == "cpu" else torch.bfloat16
    codec = load_codec_model(str(codec_path), codec_device, precision)
    print(f"  Codec loaded in {time.time() - t0:.1f}s (device: {codec_device})")
    print(f"  Sample rate: {codec.sample_rate}")

    # Decode
    print("  Decoding to audio...")
    from fish_speech.models.text2semantic.inference import decode_to_audio

    codes_device = codes.to(codec_device)
    audio = decode_to_audio(codes_device, codec)
    print(f"  Audio shape: {audio.shape}")
    print(f"  Duration: {audio.shape[0] / codec.sample_rate:.2f}s")

    # Save as WAV
    output_path = MODEL_DIR / "test_output.wav"
    try:
        import soundfile as sf
        sf.write(str(output_path), audio.cpu().float().numpy(), codec.sample_rate)
    except ImportError:
        from scipy.io import wavfile
        import numpy as np
        audio_np = audio.cpu().float().numpy()
        # Normalize to int16 range
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wavfile.write(str(output_path), codec.sample_rate, audio_int16)

    print(f"  Saved to {output_path}")
    print("  PASSED\n")


def main():
    import torch

    print(f"\nFish Audio S2 Pro - Inference Test")
    print(f"Model dir:  {MODEL_DIR}")
    print(f"Fish Speech: {FISH_SPEECH_DIR}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"CUDA:        {torch.cuda.is_available()}")
    print(f"MPS:         {torch.backends.mps.is_available()}")
    print()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    stages = [1, 2, 3, 4]
    if len(sys.argv) > 1:
        stages = [int(s) for s in sys.argv[1:]]
    print(f"Running stages: {stages}, device: {device}\n")

    tokenizer = None
    model = None
    decode_one_token = None
    codes = None

    try:
        if 1 in stages:
            tokenizer = stage1_tokenizer()

        if 2 in stages:
            model, decode_one_token = stage2_model(device)

        if 3 in stages:
            if model is None:
                model, decode_one_token = stage2_model(device)
            codes = stage3_generate(model, decode_one_token, device)

        if 4 in stages:
            if codes is None and 3 not in stages:
                print("Stage 4 requires stage 3 output. Skipping.")
            else:
                stage4_codec(codes, device)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("All requested stages completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
