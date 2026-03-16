"""
Fish Audio S2 Pro - Remote TTS CLI

Runs inference on a remote GPU server, streams audio back to play locally.

Usage:
  python3 speak_remote.py                          # default host
  python3 speak_remote.py --host 192.168.5.46      # custom host
"""

import os
import sys
import subprocess
import tempfile
import time

REMOTE_HOST = "192.168.5.46"
REMOTE_VENV = "source ~/Developer/fish/venv/bin/activate"
REMOTE_DIR = "~/Developer/fish/s2-pro"
REMOTE_SCRIPT = f"{REMOTE_DIR}/speak_server.py"


def main():
    args = sys.argv[1:]

    host = REMOTE_HOST
    if "--host" in args:
        idx = args.index("--host")
        host = args[idx + 1]

    no_save = "--no-save" in args
    out_dir = None if no_save else "output"
    if "--out" in args:
        idx = args.index("--out")
        out_dir = args[idx + 1]

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Fish Audio S2 Pro - Remote TTS ({host})\n")
    print("Loading model on remote GPU (this takes ~90s first time)...")

    # Start persistent remote process
    remote_cmd = (
        f"{REMOTE_VENV} && cd {REMOTE_DIR} && "
        f"python3 speak_server.py"
    )

    proc = subprocess.Popen(
        ["ssh", "-T", host, remote_cmd],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for "READY" signal from remote
    while True:
        line = proc.stdout.readline().decode().strip()
        if line == "READY":
            break
        elif line:
            print(f"  {line}")
        if proc.poll() is not None:
            err = proc.stderr.read().decode()
            print(f"Remote process died:\n{err}")
            sys.exit(1)

    print("\nReady! Type text to speak. Ctrl+C to quit.")
    print("Tip: use [tags] like [whisper], [excited], [laughing]\n")

    n = 0
    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                continue

            t0 = time.time()

            # Send text to remote
            proc.stdin.write((text + "\n").encode())
            proc.stdin.flush()

            # Read wav bytes back (protocol: SIZE\n<bytes>)
            size_line = proc.stdout.readline().decode().strip()
            if size_line.startswith("ERROR:"):
                print(f"  {size_line}")
                continue

            size = int(size_line)
            wav_data = b""
            while len(wav_data) < size:
                chunk = proc.stdout.read(size - len(wav_data))
                if not chunk:
                    break
                wav_data += chunk

            elapsed = time.time() - t0

            # Save and play
            n += 1
            if out_dir:
                save_path = os.path.join(out_dir, f"{n:03d}.wav")
                with open(save_path, "wb") as f:
                    f.write(wav_data)
                print(f"  saved: {save_path} ({elapsed:.1f}s)")
                subprocess.run(["afplay", save_path])
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav_data)
                    tmp = f.name
                print(f"  ({elapsed:.1f}s)")
                subprocess.run(["afplay", tmp])
                os.unlink(tmp)

            print()

    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
