# Ivrit Transcriber

A desktop application for transcribing Hebrew audio and video files. Built with Python, PySide6 (Qt), and [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper). Runs fully offline.

## Features

- Transcribe Hebrew audio and video files (MP3, WAV, MP4, MKV, etc.)
- Outputs **SRT subtitles**, **plain text**, or both
- Two model options: **Fast** (turbo) and **Accurate** (full)
- **GPU acceleration** with NVIDIA CUDA (auto-detected)
- Voice Activity Detection (VAD)
- Progress tracking with ETA
- Custom output filenames

## Requirements

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available in PATH
- CTranslate2 Hebrew Whisper models (see [Models](#models) below)
- Optional: PyTorch with CUDA for GPU acceleration

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ivrit-transcriber.git
cd ivrit-transcriber
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Models

Download the CTranslate2 Hebrew models and place them in a `Models/` directory at the project root:

```
Models/
  ivrit-large-v3-turbo-ct2/    # Fast model
  ivrit-large-v3-ct2/          # Accurate model
```

Each model folder must contain: `model.bin`, `tokenizer.json`, `vocabulary.json`.

The models are based on [ivrit.ai](https://www.ivrit.ai/) Hebrew Whisper models converted to CTranslate2 format.

## Usage

```bash
python app.py
```

1. Click **Select File** and choose an audio or video file
2. Set the output folder and options (model, format, device)
3. Click **Start Transcription**

## Building an Executable

```bash
pip install pyinstaller
pyinstaller IvritTranscriber.spec
```

The executable will be in `dist/IvritTranscriber/`.

## Project Structure

```
app.py                  # Main application and GUI
core/
  settings.py           # Settings persistence (Pydantic)
  jobs.py               # Job/Task state management
  worker.py             # Transcription worker (QRunnable)
engine/
  ffmpeg_helper.py      # FFmpeg wrapper (probe, extract, split)
  model_loader.py       # CTranslate2 model validation and loading
  gpu_detector.py       # NVIDIA CUDA detection
  transcriber.py        # Chunk transcription logic
  merger.py             # Merge chunks into final SRT/TXT
  checkpoint.py         # Checkpoint support
```

## License

MIT
