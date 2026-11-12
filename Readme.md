# Jarvis: a local voice assistant

A voice assistant that runs on your own computer. Say the wake word, ask something, and it answers out loud while a 3D particle orb reacts to what it is doing.

Everything runs locally: speech recognition, the language model and the voice. Nothing is sent to a cloud service (the orb page only loads the three.js library from a CDN).

## How it works

```
mic -> wake word -> voice activity detection -> speech-to-text -> LLM -> text-to-speech -> speaker
        (openWakeWord)   (Silero VAD)            (faster-whisper)  (Ollama)  (Piper)
```

The orb (`visual.html`) runs inside the app's own window and changes with the assistant's state:

| State     | Orb                                          |
|-----------|----------------------------------------------|
| Idle      | Slow blue breathing, waiting for wake word   |
| Listening | Cyan, pulses with your microphone level      |
| Thinking  | Violet, contracts and spins                  |
| Speaking  | Amber, ripples with the assistant's voice    |

## Requirements

- Windows 10/11 (Linux and macOS may work, but only Windows has been tried; see notes below)
- Python 3.11 (3.13 can fail to install some dependencies)
- [Ollama](https://ollama.com) installed and running
- A microphone and speakers
- NVIDIA GPU optional; the default settings run on CPU

## Setup

```powershell
# 1. Create and activate a virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Pull a language model
ollama pull llama3.1:8b

# 4. Download a Piper voice into this folder
python -m piper.download_voices en_US-lessac-medium
```

If `piper.download_voices` isn't available in your version, download both `en_US-lessac-medium.onnx` and `en_US-lessac-medium.onnx.json` from the [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) page on Hugging Face and place them next to `main.py`.

## Run

```powershell
python main.py
```

A window opens and the assistant says "Ready." Say **"hey jarvis"**, wait for the beep, then speak. It stays open for a few seconds after each reply so you can keep talking. Say "goodbye" or close the window to quit.

## Configuration

Settings are at the top of `main.py`:

| Setting | Purpose |
|---------|---------|
| `LLM_MODEL` | Ollama model name |
| `LLM_USE_GPU` | `False` forces CPU for the LLM (useful if Ollama's GPU backend crashes) |
| `WHISPER_MODEL` | Speech recognition size: `base.en` is faster, `small.en` more accurate |
| `PIPER_MODEL` | Voice file; set `PIPER_SAMPLE_RATE` to match its `.onnx.json` |
| `WAKE_WORD` | Built-in options: `hey_jarvis`, `alexa`, `hey_mycroft`, `hey_rhasspy` |
| `WAKE_THRESHOLD` | Raise if it triggers by accident, lower if it misses you |
| `SYSTEM_PROMPT` | The assistant's personality |

## Troubleshooting

- **`ModuleNotFoundError`**: you are running a different Python than the one you installed into. Activate the `.venv` and use `python -m pip install ...`.
- **Piper fails to start**: check the voice `.onnx` and `.onnx.json` files are in the project folder. The script runs Piper through `python -m piper`, so a moved virtual environment does not break it.
- **Ollama "CUDA error" or crash**: update your NVIDIA driver and Ollama, or keep `LLM_USE_GPU = False`.
- **Blank window**: make sure the Microsoft Edge WebView2 runtime is installed.

## Notes and limits

- Voice models are not included in this repo. Download them separately and respect each model's licence (Llama models use Meta's community licence).
- Do not clone a real person's voice without their consent.
- This is a hobby project and has been tested lightly.