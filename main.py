
import collections
import json
import os
import queue
import re
import subprocess
import sys
import time
import traceback
 
import numpy as np
import ollama
import openwakeword
import sounddevice as sd
import torch
import webview
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeModel
from silero_vad import VADIterator, load_silero_vad
 
# ---------- config ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_MODEL = "llama3.1:8b"
LLM_USE_GPU = False                   # set True once Ollama's GPU error is fixed
WHISPER_MODEL = "small.en"            # try "base.en" if it feels slow on CPU
PIPER_MODEL = os.path.join(BASE_DIR, "en_GB-alan-medium.onnx")
PIPER_SAMPLE_RATE = 22050             # must match the voice's .onnx.json
MIC_SAMPLE_RATE = 16000
SILENCE_MS = 700                      # silence that ends your turn
MAX_HISTORY = 12                      # messages kept besides the system prompt
 
WAKE_WORD_ENABLED = True              # False = always listening, no trigger
WAKE_WORD = "hey_jarvis"              # also built in: alexa, hey_mycroft, hey_rhasspy
WAKE_THRESHOLD = 0.5                  # raise if false triggers, lower if it misses you
FOLLOW_UP_SECONDS = 6                 # keep listening this long after a reply
FIRST_SPEECH_TIMEOUT = 8              # give up if you say nothing after the wake word
 
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Reply in one to three short, "
    "conversational sentences. No markdown, lists, or emojis, because "
    "your reply will be spoken aloud."
)
 
# ---------- models ----------
print("Loading models...")
vad_model = load_silero_vad()
stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
 
if WAKE_WORD_ENABLED:
    openwakeword.utils.download_models()  # no-op if already downloaded
    wake_model = WakeModel(wakeword_models=[WAKE_WORD], inference_framework="onnx")
 
 
# ---------- visual (runs inside the app window) ----------
_window = None
 
 
def send(state: str | None = None, level: float = 0.0) -> None:
    """Update the orb: state is idle/listening/thinking/speaking, level is 0-1."""
    if _window is None:
        return
    js = f"window.assistant && window.assistant.update({json.dumps(state)}, {round(float(level), 3)})"
    try:
        _window.evaluate_js(js)
    except Exception:
        pass  # window closing or not ready yet
 
 
# ---------- wake word ----------
def wait_for_wake_word() -> None:
    """Block until the wake word is heard. Uses 80 ms frames (1280 samples)."""
    send("idle")
    if hasattr(wake_model, "reset"):
        wake_model.reset()
    with sd.InputStream(
        samplerate=MIC_SAMPLE_RATE, channels=1, dtype="int16", blocksize=1280
    ) as stream:
        while True:
            frame, _ = stream.read(1280)
            scores = wake_model.predict(frame[:, 0])
            if scores and max(scores.values()) >= WAKE_THRESHOLD:
                return
 
 
def beep() -> None:
    """Short tone so you know it is listening."""
    t = np.linspace(0, 0.12, int(MIC_SAMPLE_RATE * 0.12), endpoint=False)
    tone = (0.25 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    sd.play(tone, MIC_SAMPLE_RATE)
    sd.wait()
 
 
# ---------- listening ----------
def listen(timeout: float | None = None) -> np.ndarray | None:
    """Block until the user finishes speaking; return mono float32 audio.
    Returns None if no speech starts within `timeout` seconds."""
    q: queue.Queue = queue.Queue()
 
    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())
 
    vad = VADIterator(
        vad_model,
        sampling_rate=MIC_SAMPLE_RATE,
        min_silence_duration_ms=SILENCE_MS,
    )
    pre_roll = collections.deque(maxlen=10)  # ~320 ms before speech starts
    buffer: list = []
    recording = False
    started = time.monotonic()
 
    # The mic is only open while listening, so the assistant never hears itself.
    with sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=512,  # Silero VAD expects 512 samples at 16 kHz
        callback=callback,
    ):
        while True:
            chunk = q.get()
            if q.qsize() < 3:  # skip visual updates if we're falling behind
                send("listening", min(1.0, float(np.sqrt(np.mean(chunk**2))) * 8))
            event = vad(torch.from_numpy(chunk))
            if not recording:
                pre_roll.append(chunk)
                if event and "start" in event:
                    recording = True
                    buffer = list(pre_roll)
                elif timeout and time.monotonic() - started > timeout:
                    return None
            else:
                buffer.append(chunk)
                if event and "end" in event:
                    vad.reset_states()
                    return np.concatenate(buffer)
 
 
def transcribe(audio: np.ndarray) -> str:
    segments, _ = stt.transcribe(audio, language="en", beam_size=1)
    return " ".join(s.text.strip() for s in segments).strip()
 
 
# ---------- speaking ----------
def speak(text: str) -> None:
    """Synthesize speech with Piper and play it, animating the orb."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "piper", "--model", PIPER_MODEL, "--output-raw"],
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("Piper failed:", e.stderr.decode(errors="replace"))
        raise
    audio = np.frombuffer(proc.stdout, dtype=np.int16)
 
    if audio.size:
        hop = int(PIPER_SAMPLE_RATE * 0.05)  # 50 ms steps for the orb
        sd.play(audio, PIPER_SAMPLE_RATE)
        t0 = time.monotonic()
        for k in range(0, audio.size, hop):
            seg = audio[k:k + hop].astype(np.float32) / 32768.0
            send("speaking", min(1.0, float(np.sqrt(np.mean(seg**2))) * 6))
            time.sleep(max(0.0, t0 + (k + hop) / PIPER_SAMPLE_RATE - time.monotonic()))
        sd.wait()
 
 
# ---------- brain ----------
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
 
 
def reply(history: list) -> str:
    """Stream the LLM response and speak it sentence by sentence."""
    full, pending = "", ""
    options = {} if LLM_USE_GPU else {"num_gpu": 0}
    stream = ollama.chat(model=LLM_MODEL, messages=history, stream=True, options=options)
    for part in stream:
        token = part["message"]["content"]
        full += token
        pending += token
        *sentences, pending = SENTENCE_END.split(pending)
        for s in sentences:
            if s.strip():
                speak(s.strip())
    if pending.strip():
        speak(pending.strip())
    return full.strip()
 
 
# ---------- assistant loop (runs in a background thread) ----------
def assistant_loop() -> None:
    try:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        speak("Ready.")
 
        while True:
            # 1. Wait for the trigger (or skip it if disabled)
            if WAKE_WORD_ENABLED:
                print(f"Waiting for wake word '{WAKE_WORD}'...")
                wait_for_wake_word()
                beep()
                send("listening")
                timeout = FIRST_SPEECH_TIMEOUT
            else:
                print("Listening...")
                timeout = None
 
            # 2. Conversation: keep going while you keep talking
            while True:
                audio = listen(timeout=timeout)
                if audio is None:  # nobody spoke, go back to waiting
                    break
                text = transcribe(audio)
                if not text:
                    timeout = FOLLOW_UP_SECONDS if WAKE_WORD_ENABLED else None
                    continue
                print(f"You: {text}")
                send("thinking")
                if "goodbye" in text.lower():
                    speak("Goodbye.")
                    _window.destroy()
                    return
 
                history.append({"role": "user", "content": text})
                answer = reply(history)
                print(f"Jarvis: {answer}")
                history.append({"role": "assistant", "content": answer})
 
                # keep the system prompt plus the most recent messages
                history = history[:1] + history[1:][-MAX_HISTORY:]
 
                # follow-up window: no need to repeat the wake word
                timeout = FOLLOW_UP_SECONDS if WAKE_WORD_ENABLED else None
    except Exception:
        traceback.print_exc()
 
 
def main() -> None:
    global _window
    with open(os.path.join(BASE_DIR, "visual.html"), encoding="utf-8") as f:
        html = f.read().replace("const EMBEDDED = false;", "const EMBEDDED = true;")
 
    _window = webview.create_window(
        "Jarvis", html=html, width=760, height=760,
        background_color="#05070d", resizable=True,
    )
    webview.start(assistant_loop)  # window runs here, assistant in a thread
    os._exit(0)  # window closed: stop the audio loop too
 
 
if __name__ == "__main__":
    main()