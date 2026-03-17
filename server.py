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
VOICES_DIR = MODEL_DIR / "voices"
sys.path.insert(0, str(FISH_SPEECH_DIR))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response, JSONResponse, FileResponse
import uvicorn

# --- Audio Save Settings ---
SETTINGS_FILE = MODEL_DIR / "settings.json"
DEFAULT_SAVE_DIR = Path.home() / "Documents" / "audio-generated"

def load_settings():
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {"save_dir": str(DEFAULT_SAVE_DIR), "auto_save": True}

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

_settings = load_settings()

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


def parse_speaker_segments(text, default_speaker=0):
    """Parse text with {Speaker} tags into [(speaker_name, text), ...] segments.

    Returns a list of (speaker_name, text_content) tuples.
    speaker_name is the lowercase resolved name (e.g. "ivy", "flint").
    """
    name_to_id = {name.lower(): sid for sid, name in _speakers.items()}
    id_to_name = {sid: name.lower() for sid, name in _speakers.items()}

    # Find all {Speaker} tags and split text around them
    pattern = r'\{([^}]+)\}'
    parts = re.split(pattern, text)

    segments = []
    current_speaker = id_to_name.get(str(default_speaker), "aria")

    i = 0
    while i < len(parts):
        if i % 2 == 0:
            # Text part
            content = parts[i].strip()
            if content:
                segments.append((current_speaker, content))
        else:
            # Speaker name part
            name = parts[i].strip()
            # Resolve to a known speaker name
            if name.lower() in name_to_id:
                current_speaker = name.lower()
            elif name.isdigit() and 0 <= int(name) < 100:
                current_speaker = id_to_name.get(name, f"voice {name}")
            else:
                current_speaker = name.lower()
        i += 1

    # If no segments found, return the whole text with default speaker
    if not segments:
        segments = [(current_speaker, text.strip())]

    return segments


# --- Voice Reference Cache ---
_voice_cache = {}  # speaker_name_lower -> (prompt_text, prompt_tokens) or None


def load_voice_reference(speaker_name):
    """Load voice reference audio for a speaker from voices/<name>/ directory.

    Returns (prompt_text, prompt_tokens) or (None, None) if no reference exists.
    """
    cache_key = speaker_name.lower()
    if cache_key in _voice_cache:
        return _voice_cache[cache_key]

    voice_dir = VOICES_DIR / cache_key
    if not voice_dir.exists():
        _voice_cache[cache_key] = (None, None)
        return None, None

    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.opus'}
    wav_files = [f for f in voice_dir.iterdir()
                 if f.suffix.lower() in audio_extensions]
    if not wav_files:
        _voice_cache[cache_key] = (None, None)
        return None, None

    wav_file = wav_files[0]
    lab_file = wav_file.with_suffix('.lab')
    prompt_text = lab_file.read_text(encoding='utf-8').strip() if lab_file.exists() else ""

    # Encode the reference audio to VQ tokens (codec is on CPU)
    from fish_speech.models.text2semantic.inference import encode_audio
    codec_device = next(_codec.parameters()).device
    prompt_tokens = encode_audio(wav_file, _codec, codec_device)

    _voice_cache[cache_key] = (prompt_text, prompt_tokens)
    return prompt_text, prompt_tokens


def invalidate_voice_cache(speaker_name=None):
    """Clear cached voice references."""
    global _voice_cache
    if speaker_name:
        _voice_cache.pop(speaker_name.lower(), None)
    else:
        _voice_cache.clear()


def get_voice_status():
    """Return dict of speaker names -> whether they have a voice reference."""
    status = {}
    for sid, name in sorted(_speakers.items(), key=lambda x: int(x[0])):
        voice_dir = VOICES_DIR / name.lower()
        has_voice = voice_dir.exists() and any(
            f.suffix.lower() in {'.wav', '.mp3', '.flac', '.ogg', '.opus'}
            for f in voice_dir.iterdir()
        ) if voice_dir.exists() else False
        status[sid] = {"name": name, "has_voice": has_voice}
    return status


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
    import numpy as np
    import soundfile as sf
    from fish_speech.models.text2semantic.inference import (
        generate_long,
        decode_to_audio,
    )

    # Parse text into per-speaker segments
    segments = parse_speaker_segments(text, default_speaker=speaker)

    # Group consecutive segments by same speaker to minimize generation calls
    grouped = []
    for spk, content in segments:
        if grouped and grouped[-1][0] == spk:
            grouped[-1] = (spk, grouped[-1][1] + " " + content)
        else:
            grouped.append((spk, content))

    all_audio_segments = []
    silence_samples = int(_codec.sample_rate * 0.15)  # 150ms silence between speakers

    for seg_idx, (spk_name, seg_text) in enumerate(grouped):
        # Load voice reference for this speaker (if available)
        prompt_text, prompt_tokens = load_voice_reference(spk_name)

        # Wrap in lists if we have reference audio (as generate_long expects)
        if prompt_tokens is not None:
            ref_texts = [prompt_text or ""]
            ref_tokens = [prompt_tokens]
        else:
            ref_texts = None
            ref_tokens = None

        # Use speaker 0 tag (identity comes from reference audio, not the tag)
        tagged_text = f"<|speaker:0|>{seg_text}"

        seg_codes = []
        for response in generate_long(
            model=_model,
            device=_device,
            decode_one_token=_decode_one_token,
            text=tagged_text,
            num_samples=1,
            max_new_tokens=0,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            compile=False,
            iterative_prompt=True,
            chunk_length=300,
            prompt_text=ref_texts,
            prompt_tokens=ref_tokens,
        ):
            if response.action == "sample":
                seg_codes.append(response.codes)
            elif response.action == "next":
                break

        if seg_codes:
            codes = torch.cat(seg_codes, dim=1)
            audio = decode_to_audio(codes.to("cpu"), _codec)
            all_audio_segments.append(audio)

            # Add silence between different speakers (not after last segment)
            if seg_idx < len(grouped) - 1:
                silence = torch.zeros(silence_samples)
                all_audio_segments.append(silence)

    if not all_audio_segments:
        return None

    final_audio = torch.cat(all_audio_segments)

    buf = io.BytesIO()
    sf.write(buf, final_audio.float().numpy(), _codec.sample_rate, format="WAV")
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

  /* Voice cloning panel */
  .voice-panel { margin-top: 16px; border-top: 1px solid #222; padding-top: 16px; }
  .voice-panel summary { cursor: pointer; color: #666; font-size: 0.85rem; }
  .voice-panel summary:hover { color: #aaa; }
  .voice-list { margin-top: 12px; }
  .voice-item { display: flex; align-items: center; gap: 8px; padding: 6px 0;
                border-bottom: 1px solid #1a1a1a; font-size: 0.85rem; }
  .voice-item .vi-name { width: 100px; color: #aaa; flex-shrink: 0; }
  .voice-item .vi-status { font-size: 0.75rem; padding: 2px 8px; border-radius: 10px; flex-shrink: 0; }
  .voice-item .vi-status.has-voice { background: #1a2a1a; color: #4a9; border: 1px solid #2a4a2a; }
  .voice-item .vi-status.no-voice { background: #1a1a1a; color: #555; border: 1px solid #333; }
  .voice-item .vi-actions { display: flex; gap: 4px; margin-left: auto; }
  .voice-item button { background: #222; border: 1px solid #333; color: #888; border-radius: 4px;
                       cursor: pointer; font-size: 0.75rem; padding: 3px 8px; }
  .voice-item button:hover { border-color: #4a9eff; color: #aaa; }
  .voice-item button.vi-delete { color: #a55; }
  .voice-item button.vi-delete:hover { border-color: #f66; color: #f66; }
  .voice-upload { margin-top: 12px; background: #111; border: 1px solid #222; border-radius: 8px;
                  padding: 12px; display: none; }
  .voice-upload label { font-size: 0.8rem; color: #888; display: block; margin-bottom: 4px; }
  .voice-upload input[type="file"] { font-size: 0.8rem; color: #aaa; margin-bottom: 8px; }
  .voice-upload input[type="text"] { width: 100%; background: #1a1a1a; border: 1px solid #333;
                                     color: #e0e0e0; padding: 6px 8px; border-radius: 4px;
                                     font-size: 0.85rem; margin-bottom: 8px; }
  .voice-upload .vu-btns { display: flex; gap: 8px; }
  .speaker-tag.has-voice { border-color: #2a4a2a; background: #0a1a0a; }
  .speaker-tag.has-voice::after { content: ' \u25CF'; color: #4a9; font-size: 0.6rem; }

  /* Download buttons */
  .dl-btn { display: inline-flex; align-items: center; gap: 4px; background: #1a2a1a;
            border: 1px solid #2a4a2a; color: #6b9; border-radius: 4px; cursor: pointer;
            font-size: 0.75rem; padding: 3px 8px; margin-left: 4px; }
  .dl-btn:hover { background: #2a3a2a; border-color: #4a9; color: #8dc; }
  .dl-btn svg { width: 12px; height: 12px; fill: currentColor; }
  .history-item .item-actions { display: flex; gap: 6px; align-items: center; margin-top: 6px; }

  /* Save settings panel */
  .save-panel { margin-top: 16px; border-top: 1px solid #222; padding-top: 16px; }
  .save-panel summary { cursor: pointer; color: #666; font-size: 0.85rem; }
  .save-panel summary:hover { color: #aaa; }
  .save-settings { margin-top: 12px; }
  .save-settings label { font-size: 0.8rem; color: #888; display: block; margin-bottom: 4px; }
  .save-settings input[type="text"] { width: 100%; background: #1a1a1a; border: 1px solid #333;
    color: #e0e0e0; padding: 6px 8px; border-radius: 4px; font-size: 0.85rem; margin-bottom: 8px; }
  .save-settings .toggle { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .save-settings .toggle input[type="checkbox"] { accent-color: #4a9eff; }
  .save-indicator { font-size: 0.75rem; color: #4a9; padding: 2px 8px; background: #0a1a0a;
                    border-radius: 10px; border: 1px solid #1a3a1a; }
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

  <details class="voice-panel">
    <summary>Voice Cloning (assign reference audio to speakers)</summary>
    <p style="font-size:0.78rem; color:#555; margin-top:8px; line-height:1.4;">
      Upload a short audio clip (5&ndash;15 sec) for each speaker to clone their voice.
      Speakers with a reference clip will use voice cloning; others use the default model voice.
    </p>
    <div class="voice-list" id="voiceList"></div>
    <div class="voice-upload" id="voiceUpload">
      <label>Uploading voice for: <strong id="vuSpeakerName"></strong></label>
      <label>Audio file (WAV, MP3, FLAC &mdash; 5-15 sec recommended)</label>
      <input type="file" id="vuFile" accept=".wav,.mp3,.flac,.ogg,.opus">
      <label>Transcript of the audio (helps quality)</label>
      <input type="text" id="vuText" placeholder="What the speaker says in the clip...">
      <div class="vu-btns">
        <button onclick="submitVoiceUpload()" style="background:#2a6a2a; color:#ccc; padding:6px 16px;">Upload</button>
        <button onclick="cancelVoiceUpload()" style="background:#333; color:#888; padding:6px 16px;">Cancel</button>
      </div>
    </div>
  </details>

  <details class="save-panel">
    <summary>Auto-Save Settings</summary>
    <div class="save-settings">
      <div class="toggle">
        <input type="checkbox" id="autoSave">
        <label for="autoSave" style="display:inline; margin:0;">Auto-save generated audio to disk</label>
      </div>
      <label>Save directory:</label>
      <input type="text" id="saveDir" placeholder="Documents/audio-generated">
      <button onclick="updateSaveSettings()" style="padding:6px 16px; font-size:0.85rem; background:#2a6a2a; color:#ccc; border:none; border-radius:4px; cursor:pointer;">
        Save Settings</button>
    </div>
  </details>
</div>

<script>
let speakers = {};
let voiceStatus = {};
let saveSettings = {};

async function loadSpeakers() {
  const resp = await fetch('/v1/speakers');
  speakers = await resp.json();
  await loadVoices();
  await loadSaveSettings();
  renderSpeakerDropdown();
  renderSpeakerTags();
  renderSpeakerGrid();
  renderVoiceList();
}

async function loadVoices() {
  try {
    const resp = await fetch('/v1/voices');
    voiceStatus = await resp.json();
  } catch(e) { voiceStatus = {}; }
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
    const hasVoice = voiceStatus[id] && voiceStatus[id].has_voice;
    span.className = 'speaker-tag' + (hasVoice ? ' has-voice' : '');
    span.textContent = name;
    span.title = hasVoice ? name + ' (voice cloned)' : name + ' (default voice)';
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
    const savedFilename = resp.headers.get('X-Saved-Filename') || '';

    const fname = savedFilename || 'speech.wav';
    const container = document.getElementById('audioContainer');
    container.innerHTML = '<audio controls autoplay src="' + url + '"></audio>';
    const dlBtn = document.createElement('button');
    dlBtn.className = 'dl-btn';
    dlBtn.innerHTML = dlIcon + ' Download';
    dlBtn.onclick = function() { downloadBlob(url, fname); };
    container.appendChild(dlBtn);
    if (savedFilename) {
      const indicator = document.createElement('span');
      indicator.className = 'save-indicator';
      indicator.textContent = 'Saved: ' + savedFilename;
      container.appendChild(document.createTextNode(' '));
      container.appendChild(indicator);
    }

    status.textContent = 'Generated in ' + elapsed + 's';

    const historySection = document.getElementById('historySection');
    historySection.style.display = 'block';
    const historyList = document.getElementById('historyList');
    const item = document.createElement('div');
    item.className = 'history-item';
    const itemText = document.createElement('div');
    itemText.className = 'text';
    itemText.textContent = text;
    const itemMeta = document.createElement('div');
    itemMeta.className = 'meta';
    itemMeta.textContent = elapsed + 's \u2014 ' + new Date().toLocaleTimeString() +
      (savedFilename ? ' \u2014 ' + savedFilename : '');
    const itemAudio = document.createElement('audio');
    itemAudio.controls = true;
    itemAudio.src = url;
    const itemActions = document.createElement('div');
    itemActions.className = 'item-actions';
    const itemDlBtn = document.createElement('button');
    itemDlBtn.className = 'dl-btn';
    itemDlBtn.innerHTML = dlIcon + ' Download';
    itemDlBtn.onclick = function() { downloadBlob(url, fname); };
    itemActions.appendChild(itemDlBtn);
    item.appendChild(itemText);
    item.appendChild(itemMeta);
    item.appendChild(itemAudio);
    item.appendChild(itemActions);
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

// --- Download helpers ---
const dlIcon = '<svg viewBox="0 0 16 16"><path d="M8 12l-4-4h2.5V2h3v6H12L8 12zm-6 2h12v1.5H2V14z"/></svg>';

function downloadBlob(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// --- Save Settings ---
async function loadSaveSettings() {
  try {
    const resp = await fetch('/v1/settings');
    saveSettings = await resp.json();
    document.getElementById('autoSave').checked = saveSettings.auto_save !== false;
    document.getElementById('saveDir').value = saveSettings.save_dir || '';
  } catch(e) {}
}

async function updateSaveSettings() {
  const autoSave = document.getElementById('autoSave').checked;
  const saveDir = document.getElementById('saveDir').value.trim();
  try {
    const resp = await fetch('/v1/settings', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ auto_save: autoSave, save_dir: saveDir || undefined })
    });
    if (resp.ok) {
      saveSettings = await resp.json();
      document.getElementById('status').textContent = 'Save settings updated!';
    }
  } catch(e) {
    document.getElementById('status').textContent = 'Error updating settings';
  }
}

// --- Voice Cloning UI ---
let _vuSpeakerName = '';

function renderVoiceList() {
  const list = document.getElementById('voiceList');
  list.innerHTML = '';
  const entries = Object.entries(voiceStatus).sort((a,b) => parseInt(a[0]) - parseInt(b[0]));
  for (const [id, info] of entries.slice(0, 20)) {
    const item = document.createElement('div');
    item.className = 'voice-item';

    const nameDiv = document.createElement('div');
    nameDiv.className = 'vi-name';
    nameDiv.textContent = info.name;

    const statusDiv = document.createElement('div');
    statusDiv.className = 'vi-status ' + (info.has_voice ? 'has-voice' : 'no-voice');
    statusDiv.textContent = info.has_voice ? 'Cloned' : 'Default';

    const actions = document.createElement('div');
    actions.className = 'vi-actions';

    if (info.has_voice) {
      const playBtn = document.createElement('button');
      playBtn.textContent = 'Preview';
      playBtn.onclick = () => previewVoice(info.name);
      actions.appendChild(playBtn);

      const delBtn = document.createElement('button');
      delBtn.className = 'vi-delete';
      delBtn.textContent = 'Remove';
      delBtn.onclick = () => deleteVoice(info.name);
      actions.appendChild(delBtn);
    }

    const uploadBtn = document.createElement('button');
    uploadBtn.textContent = info.has_voice ? 'Replace' : 'Upload';
    uploadBtn.onclick = () => showVoiceUpload(info.name);
    actions.appendChild(uploadBtn);

    item.appendChild(nameDiv);
    item.appendChild(statusDiv);
    item.appendChild(actions);
    list.appendChild(item);
  }
}

function showVoiceUpload(name) {
  _vuSpeakerName = name;
  document.getElementById('vuSpeakerName').textContent = name;
  document.getElementById('vuFile').value = '';
  document.getElementById('vuText').value = '';
  document.getElementById('voiceUpload').style.display = 'block';
}

function cancelVoiceUpload() {
  document.getElementById('voiceUpload').style.display = 'none';
}

async function submitVoiceUpload() {
  const file = document.getElementById('vuFile').files[0];
  if (!file) { alert('Please select an audio file.'); return; }

  const text = document.getElementById('vuText').value.trim();
  const formData = new FormData();
  formData.append('audio', file);
  formData.append('text', text);

  const status = document.getElementById('status');
  status.textContent = 'Uploading voice for ' + _vuSpeakerName + '...';

  try {
    const resp = await fetch('/v1/voices/' + encodeURIComponent(_vuSpeakerName.toLowerCase()), {
      method: 'POST',
      body: formData
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Upload failed');
    }
    status.textContent = 'Voice uploaded for ' + _vuSpeakerName + '!';
    cancelVoiceUpload();
    await loadVoices();
    renderVoiceList();
    renderSpeakerTags();
  } catch(e) {
    status.className = 'status error';
    status.textContent = 'Error: ' + e.message;
  }
}

async function deleteVoice(name) {
  if (!confirm('Remove voice reference for ' + name + '?')) return;
  try {
    const resp = await fetch('/v1/voices/' + encodeURIComponent(name.toLowerCase()), { method: 'DELETE' });
    if (!resp.ok) throw new Error('Failed to delete');
    document.getElementById('status').textContent = 'Voice removed for ' + name;
    await loadVoices();
    renderVoiceList();
    renderSpeakerTags();
  } catch(e) {
    document.getElementById('status').className = 'status error';
    document.getElementById('status').textContent = 'Error: ' + e.message;
  }
}

function previewVoice(name) {
  const url = '/v1/voices/' + encodeURIComponent(name.toLowerCase()) + '/audio';
  document.getElementById('audioContainer').innerHTML =
    '<audio controls autoplay src="' + url + '"></audio>';
  document.getElementById('status').textContent = 'Playing reference for ' + name;
}

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


def _auto_save_audio(wav_bytes, text):
    """Save audio to the configured save directory. Returns the saved filename or None."""
    if not _settings.get("auto_save", True):
        return None
    save_dir = Path(_settings.get("save_dir", str(DEFAULT_SAVE_DIR)))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from timestamp and text snippet
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Clean text for filename: take first 40 chars, remove special chars
    snippet = re.sub(r'[{}\[\]<>|\\/:*?"\']+', '', text[:40]).strip().replace(' ', '_')
    if not snippet:
        snippet = "speech"
    filename = f"{ts}_{snippet}.wav"
    filepath = save_dir / filename
    filepath.write_bytes(wav_bytes)
    return filename


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

    # Auto-save to disk
    saved_filename = _auto_save_audio(wav_bytes, text)
    if saved_filename:
        print(f"Generated {len(wav_bytes)} bytes in {elapsed:.1f}s, saved as {saved_filename}")
    else:
        print(f"Generated {len(wav_bytes)} bytes in {elapsed:.1f}s for: {text[:80]}")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Generation-Time": f"{elapsed:.2f}",
            "X-Saved-Filename": saved_filename or "",
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


@app.get("/v1/voices")
async def list_voices():
    """List all speakers and their voice reference status."""
    return JSONResponse(get_voice_status())


@app.post("/v1/voices/{speaker_name}")
async def upload_voice(speaker_name: str, audio: UploadFile = File(...), text: str = Form("")):
    """Upload a reference audio file for voice cloning."""
    speaker_name = speaker_name.strip()
    if not speaker_name:
        return JSONResponse({"detail": "Speaker name is required"}, status_code=400)

    # Validate audio file
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.opus'}
    ext = Path(audio.filename).suffix.lower() if audio.filename else '.wav'
    if ext not in allowed_extensions:
        return JSONResponse({"detail": f"Unsupported format. Use: {', '.join(allowed_extensions)}"}, status_code=400)

    voice_dir = VOICES_DIR / speaker_name.lower()
    voice_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing audio files
    for f in voice_dir.iterdir():
        if f.suffix.lower() in allowed_extensions or f.suffix.lower() == '.lab':
            f.unlink()

    # Save audio file
    audio_path = voice_dir / f"sample{ext}"
    content = await audio.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit
        return JSONResponse({"detail": "File too large (max 50MB)"}, status_code=400)
    audio_path.write_bytes(content)

    # Save transcript
    lab_path = voice_dir / "sample.lab"
    lab_path.write_text(text.strip(), encoding='utf-8')

    # Invalidate cache for this speaker
    invalidate_voice_cache(speaker_name)

    return JSONResponse({"status": "ok", "speaker": speaker_name, "file": audio.filename})


@app.delete("/v1/voices/{speaker_name}")
async def delete_voice(speaker_name: str):
    """Delete a voice reference."""
    import shutil
    voice_dir = VOICES_DIR / speaker_name.lower()
    if not voice_dir.exists():
        return JSONResponse({"detail": "Voice not found"}, status_code=404)

    shutil.rmtree(voice_dir)
    invalidate_voice_cache(speaker_name)
    return JSONResponse({"status": "ok", "speaker": speaker_name})


@app.get("/v1/voices/{speaker_name}/audio")
async def get_voice_audio(speaker_name: str):
    """Serve the reference audio file for preview."""
    voice_dir = VOICES_DIR / speaker_name.lower()
    if not voice_dir.exists():
        return JSONResponse({"detail": "Voice not found"}, status_code=404)

    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.opus'}
    for f in voice_dir.iterdir():
        if f.suffix.lower() in audio_extensions:
            return FileResponse(str(f), media_type="audio/wav")
    return JSONResponse({"detail": "No audio file found"}, status_code=404)


@app.get("/v1/settings")
async def get_settings():
    return JSONResponse(_settings)


@app.put("/v1/settings")
async def update_settings(body: dict):
    global _settings
    if "save_dir" in body:
        _settings["save_dir"] = str(body["save_dir"])
    if "auto_save" in body:
        _settings["auto_save"] = bool(body["auto_save"])
    save_settings(_settings)
    return JSONResponse(_settings)


@app.get("/v1/downloads")
async def list_downloads():
    """List all saved audio files."""
    save_dir = Path(_settings.get("save_dir", str(DEFAULT_SAVE_DIR)))
    if not save_dir.exists():
        return JSONResponse([])
    files = sorted(
        [f.name for f in save_dir.iterdir() if f.suffix.lower() == '.wav'],
        reverse=True
    )
    return JSONResponse(files)


@app.get("/v1/downloads/{filename}")
async def download_file(filename: str):
    """Download a specific saved audio file."""
    save_dir = Path(_settings.get("save_dir", str(DEFAULT_SAVE_DIR)))
    filepath = save_dir / filename
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"detail": "File not found"}, status_code=404)
    # Ensure the file is within the save directory (path traversal protection)
    if not filepath.resolve().parent == save_dir.resolve():
        return JSONResponse({"detail": "Invalid path"}, status_code=400)
    return FileResponse(
        str(filepath),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
