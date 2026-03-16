"""
Fish Audio S2 Pro - TTS Server

Provides both a web UI and a REST API for text-to-speech.

Usage:
  python server.py                    # default: 0.0.0.0:8880
  python server.py --port 9000        # custom port
  python server.py --host 127.0.0.1   # localhost only

API:
  POST /v1/tts
  Body: {"text": "Hello world", "speaker": 0, "temperature": 0.7, "top_p": 0.9, "top_k": 30}
  Returns: audio/wav

Web UI:
  GET /
"""

import asyncio
import io
import os
import sys
import time
import json
import threading
from pathlib import Path

MODEL_DIR = Path(__file__).parent.resolve()
FISH_SPEECH_DIR = MODEL_DIR.parent / "fish-speech"
sys.path.insert(0, str(FISH_SPEECH_DIR))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, JSONResponse
import uvicorn


class TTSRequest(BaseModel):
    text: str
    speaker: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 30

app = FastAPI(title="Fish Audio S2 Pro TTS", version="1.0.0")

# Global model state
_model = None
_decode_one_token = None
_codec = None
_device = None
_lock = threading.Lock()


def load_models():
    global _model, _decode_one_token, _codec, _device

    import torch
    from fish_speech.models.text2semantic.inference import (
        init_model,
        load_codec_model,
    )

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.bfloat16

    print(f"Loading TTS model on {_device}...")
    t0 = time.time()
    _model, _decode_one_token = init_model(
        checkpoint_path=str(MODEL_DIR),
        device=_device,
        precision=precision,
        compile=False,
    )
    print(f"TTS model loaded in {time.time() - t0:.1f}s")

    print("Loading audio codec...")
    t0 = time.time()
    _codec = load_codec_model(str(MODEL_DIR / "codec.pth"), "cpu", torch.float32)
    print(f"Codec loaded in {time.time() - t0:.1f}s")
    print(f"Server ready! GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB")


def generate_audio(text, speaker=0, temperature=0.7, top_p=0.9, top_k=30):
    import torch
    import soundfile as sf
    from fish_speech.models.text2semantic.inference import (
        generate_long,
        decode_to_audio,
    )

    if not text.strip().startswith("<|speaker:"):
        text = f"<|speaker:{speaker}|>{text}"

    all_codes = []
    for response in generate_long(
        model=_model,
        device=_device,
        decode_one_token=_decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=0,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
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
        return None

    codes = torch.cat(all_codes, dim=1)
    audio = decode_to_audio(codes.to("cpu"), _codec)

    buf = io.BytesIO()
    sf.write(buf, audio.float().numpy(), _codec.sample_rate, format="WAV")
    return buf.getvalue()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fish Audio S2 Pro</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0a0a; color: #e0e0e0; min-height: 100vh;
         display: flex; justify-content: center; padding: 40px 20px; }
  .container { max-width: 640px; width: 100%; }
  h1 { font-size: 1.5rem; margin-bottom: 8px; color: #fff; }
  .subtitle { color: #888; margin-bottom: 24px; font-size: 0.9rem; }
  textarea { width: 100%; height: 120px; padding: 12px; border: 1px solid #333;
             border-radius: 8px; background: #1a1a1a; color: #e0e0e0;
             font-size: 1rem; resize: vertical; font-family: inherit; }
  textarea:focus { outline: none; border-color: #4a9eff; }
  .controls { display: flex; gap: 12px; margin-top: 12px; align-items: center; flex-wrap: wrap; }
  .control-group { display: flex; align-items: center; gap: 4px; }
  .control-group label { font-size: 0.8rem; color: #888; }
  .control-group input, .control-group select {
    background: #1a1a1a; border: 1px solid #333; color: #e0e0e0;
    padding: 6px 8px; border-radius: 4px; font-size: 0.85rem; width: 70px; }
  button { padding: 10px 24px; border: none; border-radius: 8px; cursor: pointer;
           font-size: 1rem; font-weight: 600; transition: all 0.2s; }
  #speakBtn { background: #4a9eff; color: #fff; }
  #speakBtn:hover { background: #3a8eef; }
  #speakBtn:disabled { background: #333; color: #666; cursor: not-allowed; }
  .status { margin-top: 16px; font-size: 0.9rem; color: #888; min-height: 1.2em; }
  .status.error { color: #ff6b6b; }
  #audioContainer { margin-top: 16px; }
  audio { width: 100%; }
  .tags { margin-top: 16px; font-size: 0.8rem; color: #666; }
  .tags span { display: inline-block; background: #1a1a1a; border: 1px solid #333;
               padding: 2px 8px; border-radius: 12px; margin: 2px; cursor: pointer; }
  .tags span:hover { border-color: #4a9eff; color: #aaa; }
  .history { margin-top: 24px; }
  .history h3 { font-size: 0.9rem; color: #888; margin-bottom: 8px; }
  .history-item { background: #1a1a1a; border: 1px solid #222; border-radius: 8px;
                  padding: 12px; margin-bottom: 8px; }
  .history-item .text { font-size: 0.85rem; color: #aaa; margin-bottom: 8px; }
  .history-item .meta { font-size: 0.75rem; color: #555; }
  .history-item audio { width: 100%; margin-top: 8px; }
</style>
</head>
<body>
<div class="container">
  <h1>Fish Audio S2 Pro</h1>
  <p class="subtitle">Text-to-Speech &mdash; RTX 5090 local inference</p>

  <textarea id="text" placeholder="Type text to speak... Use [tags] like [whisper], [excited], [laughing] for control."></textarea>

  <div class="controls">
    <button id="speakBtn" onclick="speak()">Speak</button>
    <div class="control-group">
      <label>Speaker</label>
      <input type="number" id="speaker" value="0" min="0" max="99">
    </div>
    <div class="control-group">
      <label>Temp</label>
      <input type="number" id="temperature" value="0.7" min="0.1" max="2.0" step="0.1">
    </div>
    <div class="control-group">
      <label>Top-P</label>
      <input type="number" id="top_p" value="0.9" min="0.1" max="1.0" step="0.05">
    </div>
  </div>

  <div class="tags">
    Common tags:
    <span onclick="insertTag('[whisper]')">[whisper]</span>
    <span onclick="insertTag('[excited]')">[excited]</span>
    <span onclick="insertTag('[laughing]')">[laughing]</span>
    <span onclick="insertTag('[pause]')">[pause]</span>
    <span onclick="insertTag('[singing]')">[singing]</span>
    <span onclick="insertTag('[angry]')">[angry]</span>
    <span onclick="insertTag('[sad]')">[sad]</span>
    <span onclick="insertTag('[shouting]')">[shouting]</span>
    <span onclick="insertTag('[emphasis]')">[emphasis]</span>
    <span onclick="insertTag('[low voice]')">[low voice]</span>
  </div>

  <div id="status" class="status"></div>
  <div id="audioContainer"></div>

  <div class="history" id="historySection" style="display:none">
    <h3>History</h3>
    <div id="historyList"></div>
  </div>
</div>

<script>
function insertTag(tag) {
  const ta = document.getElementById('text');
  const start = ta.selectionStart;
  ta.value = ta.value.slice(0, start) + tag + ta.value.slice(ta.selectionEnd);
  ta.focus();
  ta.selectionStart = ta.selectionEnd = start + tag.length;
}

async function speak() {
  const text = document.getElementById('text').value.trim();
  if (!text) return;

  const btn = document.getElementById('speakBtn');
  const status = document.getElementById('status');
  btn.disabled = true;
  btn.textContent = 'Generating...';
  status.className = 'status';
  status.textContent = 'Generating speech...';

  const t0 = performance.now();

  try {
    const resp = await fetch('/v1/tts', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        text: text,
        speaker: parseInt(document.getElementById('speaker').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('top_p').value),
      })
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Generation failed');
    }

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    // Main player
    document.getElementById('audioContainer').innerHTML =
      '<audio controls autoplay src="' + url + '"></audio>';

    status.textContent = 'Generated in ' + elapsed + 's';

    // Add to history
    const historySection = document.getElementById('historySection');
    historySection.style.display = 'block';
    const historyList = document.getElementById('historyList');
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML =
      '<div class="text">' + text.replace(/</g, '&lt;') + '</div>' +
      '<div class="meta">' + elapsed + 's &mdash; ' + new Date().toLocaleTimeString() + '</div>' +
      '<audio controls src="' + url + '"></audio>';
    historyList.insertBefore(item, historyList.firstChild);

  } catch (e) {
    status.className = 'status error';
    status.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Speak';
  }
}

document.getElementById('text').addEventListener('keydown', function(e) {
  if (e.ctrlKey && e.key === 'Enter') speak();
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return HTML_PAGE


@app.post("/v1/tts")
async def tts_api(body: TTSRequest):
    text = body.text.strip()
    if not text:
        return JSONResponse({"detail": "text is required"}, status_code=400)

    def _run():
        with _lock:
            return generate_audio(text, body.speaker, body.temperature, body.top_p, body.top_k)

    t0 = time.time()
    wav_bytes = await asyncio.to_thread(_run)
    elapsed = time.time() - t0

    if wav_bytes is None:
        return JSONResponse({"detail": "No audio generated"}, status_code=500)

    print(f"Generated {len(wav_bytes)} bytes in {elapsed:.1f}s for: {text[:80]}")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Generation-Time": f"{elapsed:.2f}",
            "Content-Disposition": 'inline; filename="speech.wav"',
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None, "device": str(_device)}


def main():
    args = sys.argv[1:]
    host = "0.0.0.0"
    port = 8880

    if "--host" in args:
        host = args[args.index("--host") + 1]
    if "--port" in args:
        port = int(args[args.index("--port") + 1])

    load_models()

    print(f"\nServer starting on http://{host}:{port}")
    print(f"Web UI: http://{host}:{port}/")
    print(f"API:    POST http://{host}:{port}/v1/tts")
    print(f"Health: GET  http://{host}:{port}/health\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
