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

  Multi-speaker:
  Body: {"text": "{Alice} Hello! {Bob} How are you?", "temperature": 0.7}

Web UI:
  GET /
"""

import asyncio
import io
import os
import re
import sys
import time
import json
import threading
from pathlib import Path
from typing import Optional

MODEL_DIR = Path(__file__).parent.resolve()
FISH_SPEECH_DIR = MODEL_DIR.parent / "fish-speech"
sys.path.insert(0, str(FISH_SPEECH_DIR))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, JSONResponse
import uvicorn

# --- Speaker Registry ---
SPEAKERS_FILE = MODEL_DIR / "speakers.json"

DEFAULT_SPEAKERS = {
    "0": "Aria",
    "1": "River",
    "2": "Nova",
    "3": "Sage",
    "4": "Echo",
    "5": "Luna",
    "6": "Atlas",
    "7": "Coral",
    "8": "Jasper",
    "9": "Willow",
    "10": "Phoenix",
    "11": "Maple",
    "12": "Orion",
    "13": "Ivy",
    "14": "Flint",
    "15": "Dune",
    "16": "Breeze",
    "17": "Cedar",
    "18": "Pearl",
    "19": "Storm",
}
# Fill remaining IDs with generic names
for i in range(20, 100):
    DEFAULT_SPEAKERS[str(i)] = f"Voice {i}"


def load_speakers():
    if SPEAKERS_FILE.exists():
        with open(SPEAKERS_FILE, "r") as f:
            return json.load(f)
    return dict(DEFAULT_SPEAKERS)


def save_speakers(speakers):
    with open(SPEAKERS_FILE, "w") as f:
        json.dump(speakers, f, indent=2)


_speakers = load_speakers()


def resolve_speaker_text(text):
    """Convert {Speaker Name} tags to <|speaker:N|> tags.

    Supports: {Alice} Hello {Bob} How are you?
    Also passes through raw <|speaker:N|> tags unchanged.
    """
    name_to_id = {name.lower(): sid for sid, name in _speakers.items()}

    def replace_match(m):
        name = m.group(1).strip()
        # Check by name (case-insensitive)
        sid = name_to_id.get(name.lower())
        if sid is not None:
            return f"<|speaker:{sid}|>"
        # Check if it's a raw numeric ID
        if name.isdigit() and 0 <= int(name) < 100:
            return f"<|speaker:{name}|>"
        # Unknown speaker name - default to speaker 0
        return f"<|speaker:0|>"

    return re.sub(r'\{([^}]+)\}', replace_match, text)


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

    # Resolve {Speaker Name} tags to <|speaker:N|> tags
    text = resolve_speaker_text(text)

    # If no speaker tag present, prepend default
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
  .container { max-width: 700px; width: 100%; }
  h1 { font-size: 1.5rem; margin-bottom: 8px; color: #fff; }
  .subtitle { color: #888; margin-bottom: 24px; font-size: 0.9rem; }
  textarea { width: 100%; height: 140px; padding: 12px; border: 1px solid #333;
             border-radius: 8px; background: #1a1a1a; color: #e0e0e0;
             font-size: 1rem; resize: vertical; font-family: inherit; }
  textarea:focus { outline: none; border-color: #4a9eff; }
  .controls { display: flex; gap: 12px; margin-top: 12px; align-items: center; flex-wrap: wrap; }
  .control-group { display: flex; align-items: center; gap: 4px; }
  .control-group label { font-size: 0.8rem; color: #888; }
  .control-group input, .control-group select {
    background: #1a1a1a; border: 1px solid #333; color: #e0e0e0;
    padding: 6px 8px; border-radius: 4px; font-size: 0.85rem; }
  .control-group input { width: 70px; }
  .control-group select { width: 140px; }
  button { padding: 10px 24px; border: none; border-radius: 8px; cursor: pointer;
           font-size: 1rem; font-weight: 600; transition: all 0.2s; }
  #speakBtn { background: #4a9eff; color: #fff; }
  #speakBtn:hover { background: #3a8eef; }
  #speakBtn:disabled { background: #333; color: #666; cursor: not-allowed; }
  .status { margin-top: 16px; font-size: 0.9rem; color: #888; min-height: 1.2em; }
  .status.error { color: #ff6b6b; }
  #audioContainer { margin-top: 16px; }
  audio { width: 100%; }

  .speaker-tags { margin-top: 12px; font-size: 0.8rem; color: #666; }
  .speaker-tags .label { margin-bottom: 4px; }
  .speaker-tag { display: inline-block; background: #1a1a1a; border: 1px solid #333;
                 padding: 3px 10px; border-radius: 12px; margin: 2px; cursor: pointer;
                 font-size: 0.8rem; }
  .speaker-tag:hover { border-color: #4a9eff; color: #aaa; }
  .speaker-tag.active { border-color: #4a9eff; background: #1a2a3a; color: #4a9eff; }

  .tags { margin-top: 12px; font-size: 0.8rem; color: #666; }
  .tags span { display: inline-block; background: #1a1a1a; border: 1px solid #333;
               padding: 2px 8px; border-radius: 12px; margin: 2px; cursor: pointer; }
  .tags span:hover { border-color: #4a9eff; color: #aaa; }

  .history { margin-top: 24px; }
  .history h3 { font-size: 0.9rem; color: #888; margin-bottom: 8px; }
  .history-item { background: #1a1a1a; border: 1px solid #222; border-radius: 8px;
                  padding: 12px; margin-bottom: 8px; }
  .history-item .text { font-size: 0.85rem; color: #aaa; margin-bottom: 8px;
                        white-space: pre-wrap; }
  .history-item .meta { font-size: 0.75rem; color: #555; }
  .history-item audio { width: 100%; margin-top: 8px; }

  .hint { margin-top: 8px; font-size: 0.78rem; color: #555; line-height: 1.5; }
  .hint code { background: #1a1a1a; padding: 1px 5px; border-radius: 3px; color: #888; }

  /* Speaker management panel */
  .speaker-panel { margin-top: 20px; border-top: 1px solid #222; padding-top: 16px; }
  .speaker-panel summary { cursor: pointer; color: #666; font-size: 0.85rem; }
  .speaker-panel summary:hover { color: #aaa; }
  .speaker-grid { display: grid; grid-template-columns: 60px 1fr 60px; gap: 4px;
                  margin-top: 12px; max-height: 300px; overflow-y: auto; padding-right: 4px; }
  .speaker-grid .sg-header { font-size: 0.75rem; color: #555; padding: 4px 0; }
  .speaker-grid input { background: #1a1a1a; border: 1px solid #333; color: #e0e0e0;
                        padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; }
  .speaker-grid .sg-id { font-size: 0.8rem; color: #666; padding: 6px 0; }
  .speaker-grid .sg-test { background: #222; border: 1px solid #333; color: #888;
                           border-radius: 4px; cursor: pointer; font-size: 0.75rem;
                           padding: 4px; }
  .speaker-grid .sg-test:hover { border-color: #4a9eff; color: #aaa; }
</style>
</head>
<body>
<div class="container">
  <div style="display:flex; justify-content:space-between; align-items:baseline;">
    <h1>Fish Audio S2 Pro</h1>
    <a href="/help" style="color:#4a9eff; font-size:0.85rem; text-decoration:none;">Help &amp; API docs</a>
  </div>
  <p class="subtitle">Text-to-Speech &mdash; multi-speaker support</p>

  <textarea id="text" placeholder="{Aria} Hello there! {River} How are you today?&#10;&#10;Or just type plain text for single-speaker mode."></textarea>

  <div class="speaker-tags" id="speakerTagsSection">
    <div class="label">Insert speaker:</div>
    <div id="speakerTags"></div>
  </div>

  <div class="controls">
    <button id="speakBtn" onclick="speak()">Speak</button>
    <div class="control-group">
      <label>Default</label>
      <select id="speaker"></select>
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

  <div class="hint">
    Multi-speaker: <code>{Aria} Hello! {River} Hi there!</code> &nbsp;|&nbsp;
    Tags: <code>[whisper]</code> <code>[excited]</code> &nbsp;|&nbsp; Ctrl+Enter to generate
  </div>

  <div class="tags">
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

  <details class="speaker-panel">
    <summary>Speaker Management (rename voices)</summary>
    <div class="speaker-grid" id="speakerGrid">
      <div class="sg-header">ID</div><div class="sg-header">Name</div><div class="sg-header"></div>
    </div>
    <div style="margin-top: 8px; display: flex; gap: 8px;">
      <button onclick="saveSpeakers()" style="padding:6px 16px; font-size:0.85rem; background:#2a6a2a; color:#ccc;">
        Save Names</button>
      <button onclick="resetSpeakers()" style="padding:6px 16px; font-size:0.85rem; background:#333; color:#888;">
        Reset to Defaults</button>
    </div>
  </details>
</div>

<script>
let speakers = {};

async function loadSpeakers() {
  const resp = await fetch('/v1/speakers');
  speakers = await resp.json();
  renderSpeakerDropdown();
  renderSpeakerTags();
  renderSpeakerGrid();
}

function renderSpeakerDropdown() {
  const sel = document.getElementById('speaker');
  sel.innerHTML = '';
  // Show first 20 by default in dropdown
  const entries = Object.entries(speakers).sort((a,b) => parseInt(a[0]) - parseInt(b[0]));
  for (const [id, name] of entries.slice(0, 20)) {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = name + ' (#' + id + ')';
    sel.appendChild(opt);
  }
}

function renderSpeakerTags() {
  const container = document.getElementById('speakerTags');
  container.innerHTML = '';
  const entries = Object.entries(speakers).sort((a,b) => parseInt(a[0]) - parseInt(b[0]));
  for (const [id, name] of entries.slice(0, 20)) {
    const span = document.createElement('span');
    span.className = 'speaker-tag';
    span.textContent = name;
    span.onclick = () => insertSpeaker(name);
    container.appendChild(span);
  }
}

function renderSpeakerGrid() {
  const grid = document.getElementById('speakerGrid');
  grid.innerHTML = '<div class="sg-header">ID</div><div class="sg-header">Name</div><div class="sg-header"></div>';
  const entries = Object.entries(speakers).sort((a,b) => parseInt(a[0]) - parseInt(b[0]));
  for (const [id, name] of entries.slice(0, 30)) {
    const idDiv = document.createElement('div');
    idDiv.className = 'sg-id';
    idDiv.textContent = '#' + id;

    const input = document.createElement('input');
    input.value = name;
    input.dataset.id = id;
    input.className = 'sg-name';

    const testBtn = document.createElement('button');
    testBtn.className = 'sg-test';
    testBtn.textContent = 'Test';
    testBtn.onclick = () => testSpeaker(id);

    grid.appendChild(idDiv);
    grid.appendChild(input);
    grid.appendChild(testBtn);
  }
}

function insertSpeaker(name) {
  const ta = document.getElementById('text');
  const start = ta.selectionStart;
  const tag = '{' + name + '} ';
  ta.value = ta.value.slice(0, start) + tag + ta.value.slice(ta.selectionEnd);
  ta.focus();
  ta.selectionStart = ta.selectionEnd = start + tag.length;
}

function insertTag(tag) {
  const ta = document.getElementById('text');
  const start = ta.selectionStart;
  ta.value = ta.value.slice(0, start) + tag + ta.value.slice(ta.selectionEnd);
  ta.focus();
  ta.selectionStart = ta.selectionEnd = start + tag.length;
}

async function saveSpeakers() {
  const inputs = document.querySelectorAll('.sg-name');
  const updated = {};
  inputs.forEach(inp => { updated[inp.dataset.id] = inp.value; });
  // Merge with existing speakers (for IDs not shown in grid)
  const merged = Object.assign({}, speakers, updated);
  const resp = await fetch('/v1/speakers', {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(merged)
  });
  if (resp.ok) {
    speakers = await resp.json();
    renderSpeakerDropdown();
    renderSpeakerTags();
    document.getElementById('status').textContent = 'Speaker names saved!';
  }
}

async function resetSpeakers() {
  const resp = await fetch('/v1/speakers/reset', { method: 'POST' });
  if (resp.ok) {
    speakers = await resp.json();
    renderSpeakerDropdown();
    renderSpeakerTags();
    renderSpeakerGrid();
    document.getElementById('status').textContent = 'Speaker names reset to defaults.';
  }
}

async function testSpeaker(id) {
  const name = speakers[id] || 'Voice ' + id;
  const status = document.getElementById('status');
  status.textContent = 'Testing ' + name + '...';

  try {
    const resp = await fetch('/v1/tts', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        text: 'Hello, my name is ' + name + '. Nice to meet you!',
        speaker: parseInt(id),
        temperature: 0.7, top_p: 0.9
      })
    });
    if (!resp.ok) throw new Error('Failed');
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('audioContainer').innerHTML =
      '<audio controls autoplay src="' + url + '"></audio>';
    status.textContent = name + ' (#' + id + ')';
  } catch(e) {
    status.className = 'status error';
    status.textContent = 'Error testing speaker';
  }
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

    document.getElementById('audioContainer').innerHTML =
      '<audio controls autoplay src="' + url + '"></audio>';

    status.textContent = 'Generated in ' + elapsed + 's';

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

loadSpeakers();
</script>
</body>
</html>"""


HELP_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Help &mdash; Fish Audio S2 Pro</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0a0a; color: #d0d0d0; min-height: 100vh;
         display: flex; justify-content: center; padding: 40px 20px; line-height: 1.65; }
  .container { max-width: 760px; width: 100%; }
  a { color: #4a9eff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  h1 { font-size: 1.6rem; margin-bottom: 4px; color: #fff; }
  .subtitle { color: #666; margin-bottom: 32px; font-size: 0.9rem; }
  h2 { font-size: 1.15rem; color: #fff; margin: 32px 0 12px; padding-bottom: 6px;
       border-bottom: 1px solid #222; }
  h3 { font-size: 0.95rem; color: #ccc; margin: 20px 0 8px; }
  p { margin-bottom: 12px; font-size: 0.92rem; }
  code { background: #1a1a1a; padding: 2px 6px; border-radius: 4px; font-size: 0.88rem;
         color: #e8c872; font-family: 'Cascadia Code', 'Fira Code', monospace; }
  pre { background: #111; border: 1px solid #222; border-radius: 8px; padding: 16px;
        overflow-x: auto; margin: 12px 0 16px; font-size: 0.85rem; line-height: 1.5; }
  pre code { background: none; padding: 0; color: #ccc; }
  .example { background: #0d1117; border: 1px solid #1a2233; border-radius: 8px;
             padding: 14px 16px; margin: 10px 0 16px; }
  .example .label { font-size: 0.78rem; color: #4a9eff; text-transform: uppercase;
                    letter-spacing: 0.5px; margin-bottom: 6px; }
  .example code { color: #e0e0e0; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0 16px; font-size: 0.88rem; }
  th { text-align: left; color: #888; font-weight: 600; padding: 8px 12px;
       border-bottom: 1px solid #333; font-size: 0.8rem; text-transform: uppercase;
       letter-spacing: 0.3px; }
  td { padding: 8px 12px; border-bottom: 1px solid #1a1a1a; }
  td code { font-size: 0.82rem; }
  .tag-grid { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 16px; }
  .tag-pill { background: #1a1a1a; border: 1px solid #333; padding: 4px 12px;
              border-radius: 14px; font-size: 0.82rem; color: #aaa; }
  .back { display: inline-block; margin-bottom: 20px; font-size: 0.85rem; }
  .nav { display: flex; gap: 16px; margin-bottom: 24px; font-size: 0.85rem;
         padding: 10px 0; border-bottom: 1px solid #222; }
  .section-note { background: #1a1a0a; border-left: 3px solid #e8c872; padding: 10px 14px;
                  margin: 12px 0; font-size: 0.85rem; color: #bbb; border-radius: 0 6px 6px 0; }
</style>
</head>
<body>
<div class="container">
  <a href="/" class="back">&larr; Back to app</a>
  <h1>Fish Audio S2 Pro</h1>
  <p class="subtitle">Text-to-Speech Server &mdash; Help &amp; API Reference</p>

  <nav class="nav">
    <a href="#quick-start">Quick Start</a>
    <a href="#multi-speaker">Multi-Speaker</a>
    <a href="#tags">Voice Tags</a>
    <a href="#speakers">Speakers</a>
    <a href="#api">API Reference</a>
    <a href="#tips">Tips</a>
  </nav>

  <!-- QUICK START -->
  <h2 id="quick-start">Quick Start</h2>
  <p>Type text in the main text area and click <strong>Speak</strong> (or press <code>Ctrl+Enter</code>).
     The selected default speaker will be used unless you specify speakers inline.</p>

  <div class="example">
    <div class="label">Single speaker</div>
    <code>Hello! This is a test of the text-to-speech system.</code>
  </div>

  <div class="example">
    <div class="label">Single speaker with emotion</div>
    <code>[excited] Wow, this sounds amazing! [pause] Don't you think?</code>
  </div>

  <!-- MULTI-SPEAKER -->
  <h2 id="multi-speaker">Multi-Speaker Mode</h2>
  <p>Use curly braces with a speaker name to switch voices mid-text.
     Each <code>{Name}</code> tag sets the voice for the text that follows it.</p>

  <div class="example">
    <div class="label">Two speakers</div>
    <code>{Aria} Hello there! How are you doing today? {River} I'm doing great, thanks for asking!</code>
  </div>

  <div class="example">
    <div class="label">Multi-speaker dialogue with tags</div>
    <code>{Nova} [whisper] Did you hear that? {Sage} [excited] Yes! It was incredible! {Nova} [laughing] I know, right?</code>
  </div>

  <div class="example">
    <div class="label">Using numeric IDs directly</div>
    <code>{0} Hello from speaker zero. {5} And hello from speaker five.</code>
  </div>

  <div class="section-note">
    Speaker names are case-insensitive: <code>{aria}</code>, <code>{Aria}</code>, and <code>{ARIA}</code> all work.
    If a name isn't recognized, it falls back to speaker 0.
  </div>

  <!-- VOICE TAGS -->
  <h2 id="tags">Voice Control Tags</h2>
  <p>Place tags anywhere in your text to control emotion, pacing, and delivery style.
     Tags can be combined and placed inline with the text.</p>

  <h3>Common Tags</h3>
  <div class="tag-grid">
    <span class="tag-pill">[whisper]</span>
    <span class="tag-pill">[excited]</span>
    <span class="tag-pill">[laughing]</span>
    <span class="tag-pill">[pause]</span>
    <span class="tag-pill">[singing]</span>
    <span class="tag-pill">[angry]</span>
    <span class="tag-pill">[sad]</span>
    <span class="tag-pill">[shouting]</span>
    <span class="tag-pill">[emphasis]</span>
    <span class="tag-pill">[low voice]</span>
  </div>

  <h3>Additional Tags</h3>
  <div class="tag-grid">
    <span class="tag-pill">[sigh]</span>
    <span class="tag-pill">[gasp]</span>
    <span class="tag-pill">[cough]</span>
    <span class="tag-pill">[clearing throat]</span>
    <span class="tag-pill">[sobbing]</span>
    <span class="tag-pill">[giggling]</span>
    <span class="tag-pill">[screaming]</span>
    <span class="tag-pill">[whispering]</span>
    <span class="tag-pill">[nervous]</span>
    <span class="tag-pill">[sarcastic]</span>
    <span class="tag-pill">[cheerful]</span>
    <span class="tag-pill">[serious]</span>
    <span class="tag-pill">[dramatic]</span>
    <span class="tag-pill">[monotone]</span>
    <span class="tag-pill">[fast]</span>
    <span class="tag-pill">[slow]</span>
  </div>

  <div class="section-note">
    The model was trained on 15,000+ unique tags. Experiment freely &mdash; many natural descriptions
    work even if not listed here (e.g. <code>[tired]</code>, <code>[robotic]</code>, <code>[storytelling]</code>).
  </div>

  <!-- SPEAKERS -->
  <h2 id="speakers">Default Speakers</h2>
  <p>The model has 100 built-in speaker slots (IDs 0&ndash;99). The first 20 have been given
     placeholder names. Use the <strong>Speaker Management</strong> panel on the main page to
     rename them after you've tested what each one sounds like.</p>

  <table>
    <tr><th>ID</th><th>Default Name</th><th>ID</th><th>Default Name</th></tr>
    <tr><td>0</td><td>Aria</td><td>10</td><td>Phoenix</td></tr>
    <tr><td>1</td><td>River</td><td>11</td><td>Maple</td></tr>
    <tr><td>2</td><td>Nova</td><td>12</td><td>Orion</td></tr>
    <tr><td>3</td><td>Sage</td><td>13</td><td>Ivy</td></tr>
    <tr><td>4</td><td>Echo</td><td>14</td><td>Flint</td></tr>
    <tr><td>5</td><td>Luna</td><td>15</td><td>Dune</td></tr>
    <tr><td>6</td><td>Atlas</td><td>16</td><td>Breeze</td></tr>
    <tr><td>7</td><td>Coral</td><td>17</td><td>Cedar</td></tr>
    <tr><td>8</td><td>Jasper</td><td>18</td><td>Pearl</td></tr>
    <tr><td>9</td><td>Willow</td><td>19</td><td>Storm</td></tr>
  </table>

  <p>Speaker names are saved to <code>speakers.json</code> and persist across server restarts.
     IDs 20&ndash;99 default to &ldquo;Voice N&rdquo;.</p>

  <!-- API REFERENCE -->
  <h2 id="api">API Reference</h2>

  <h3>POST /v1/tts</h3>
  <p>Generate speech from text. Returns <code>audio/wav</code>.</p>
  <table>
    <tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr>
    <tr><td><code>text</code></td><td>string</td><td><em>required</em></td><td>Text to speak. Supports <code>{Name}</code> tags and <code>[emotion]</code> tags.</td></tr>
    <tr><td><code>speaker</code></td><td>int</td><td>0</td><td>Default speaker ID (0&ndash;99). Used when no <code>{Name}</code> tag is present.</td></tr>
    <tr><td><code>temperature</code></td><td>float</td><td>0.7</td><td>Sampling temperature (0.1&ndash;2.0). Higher = more varied.</td></tr>
    <tr><td><code>top_p</code></td><td>float</td><td>0.9</td><td>Nucleus sampling threshold (0.1&ndash;1.0).</td></tr>
    <tr><td><code>top_k</code></td><td>int</td><td>30</td><td>Top-K sampling limit.</td></tr>
  </table>

  <pre><code># Single speaker
curl -X POST http://localhost:8880/v1/tts \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Hello world!", "speaker": 0}' \\
  -o output.wav

# Multi-speaker
curl -X POST http://localhost:8880/v1/tts \\
  -H "Content-Type: application/json" \\
  -d '{"text": "{Aria} Hi! {River} Hey there!"}' \\
  -o dialogue.wav</code></pre>

  <p>Response headers include <code>X-Generation-Time</code> with elapsed seconds.</p>

  <h3>GET /v1/speakers</h3>
  <p>Returns the current speaker name mapping as JSON (<code>{"0": "Aria", "1": "River", ...}</code>).</p>

  <h3>PUT /v1/speakers</h3>
  <p>Update speaker names. Send a JSON object of ID&rarr;name pairs. Returns the updated map.</p>
  <pre><code>curl -X PUT http://localhost:8880/v1/speakers \\
  -H "Content-Type: application/json" \\
  -d '{"0": "Alice", "1": "Bob"}'</code></pre>

  <h3>POST /v1/speakers/reset</h3>
  <p>Reset all speaker names to defaults.</p>

  <h3>GET /health</h3>
  <p>Returns server status: <code>{"status": "ok", "model_loaded": true, "device": "cuda"}</code></p>

  <!-- TIPS -->
  <h2 id="tips">Tips &amp; Notes</h2>

  <h3>Generation Parameters</h3>
  <p><strong>Temperature</strong> controls randomness. Lower values (0.3&ndash;0.5) give more consistent,
     predictable output. Higher values (0.8&ndash;1.2) add variety and expressiveness but may
     introduce artifacts. The default 0.7 is a good balance.</p>
  <p><strong>Top-P</strong> (nucleus sampling) limits the token pool. Lower values constrain output;
     0.9 works well for most cases.</p>

  <h3>Best Practices</h3>
  <p>
    &bull; Use punctuation naturally &mdash; commas, periods, and question marks guide pacing and intonation.<br>
    &bull; Place <code>[pause]</code> tags for intentional breaks between sentences.<br>
    &bull; Test speakers with the <strong>Test</strong> button in Speaker Management before using them in dialogues.<br>
    &bull; For long texts, the model processes in chunks of ~300 bytes. Very long inputs work but take proportionally longer.<br>
    &bull; Different speakers may respond differently to emotion tags &mdash; experiment to find the best combinations.
  </p>

  <h3>Model Details</h3>
  <p>Fish Audio S2 Pro uses a Dual Autoregressive architecture (4B slow AR + 400M fast AR)
     trained on 10M+ hours of audio across 80+ languages. Audio is encoded with a 10-codebook
     RVQ codec at ~21 Hz frame rate.</p>

  <div style="margin-top: 40px; padding-top: 16px; border-top: 1px solid #222;
              font-size: 0.8rem; color: #444; text-align: center;">
    Fish Audio S2 Pro &mdash; Local TTS Server
  </div>
</div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return HTML_PAGE


@app.get("/help", response_class=HTMLResponse)
async def help_page():
    return HELP_PAGE


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


@app.get("/v1/speakers")
async def get_speakers():
    return JSONResponse(_speakers)


@app.put("/v1/speakers")
async def update_speakers(body: dict):
    global _speakers
    _speakers.update(body)
    save_speakers(_speakers)
    return JSONResponse(_speakers)


@app.post("/v1/speakers/reset")
async def reset_speakers():
    global _speakers
    _speakers = dict(DEFAULT_SPEAKERS)
    save_speakers(_speakers)
    return JSONResponse(_speakers)


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
