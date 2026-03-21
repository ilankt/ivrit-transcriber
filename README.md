# Ivrit Transcriber

A desktop application for transcribing Hebrew audio and video files. Built with Python, PySide6 (Qt), and [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper). Runs fully offline.

## Features

- Transcribe Hebrew audio and video files (MP3, WAV, MP4, MKV, etc.)
- **Live transcription** — capture system audio (WASAPI loopback) with word-by-word streaming captions
- Outputs **SRT subtitles**, **plain text**, or both
- Two model options: **Fast** (turbo) and **Accurate** (full)
- **GPU acceleration** — NVIDIA CUDA and AMD Vulkan (auto-detected)
- **Dark / Light / System theme** support
- Voice Activity Detection (VAD)
- Progress tracking with ETA
- Custom output filenames

## Requirements

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available in PATH
- Hebrew Whisper models (see [Models](#models) below)
- Optional: PyTorch with CUDA for NVIDIA GPU acceleration
- Optional: whisper.cpp with Vulkan for AMD GPU acceleration (see [AMD GPU Setup](#amd-gpu-setup))

## Installation

```bash
git clone https://github.com/ilankt/ivrit-transcriber.git
cd ivrit-transcriber
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Models

The app uses [ivrit.ai](https://www.ivrit.ai/) Hebrew Whisper models. Which format you need depends on your setup:

### For CPU or NVIDIA GPU (CTranslate2 format)

Download and place in `Models/`:

```
Models/
  ivrit-large-v3-turbo-ct2/    # Fast model
  ivrit-large-v3-ct2/          # Accurate model
```

Each folder must contain: `model.bin`, `tokenizer.json`, `vocabulary.json`.

### For AMD GPU (GGML format)

Download and place in `Models/`:

```bash
# Fast model (1.5 GB)
huggingface-cli download ivrit-ai/whisper-large-v3-turbo-ggml ggml-model.bin --local-dir Models/tmp-turbo
mv Models/tmp-turbo/ggml-model.bin Models/ggml-ivrit-large-v3-turbo.bin

# Accurate model (2.9 GB)
huggingface-cli download ivrit-ai/whisper-large-v3-ggml ggml-model.bin --local-dir Models/tmp-full
mv Models/tmp-full/ggml-model.bin Models/ggml-ivrit-large-v3.bin
```

## AMD GPU Setup

AMD GPUs are supported via [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with Vulkan. The app auto-detects AMD GPUs and uses whisper.cpp when selected.

### Prerequisites

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) (select "Desktop development with C++")
2. Install [CMake](https://cmake.org/download/)
3. Install [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) (default options are fine)

### Build whisper.cpp

```bash
git clone --depth 1 --branch v1.8.4 https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j
```

### Install

Copy the built binaries into the project:

```bash
mkdir Binaries
cp build/bin/Release/whisper-cli.exe Binaries/
cp build/bin/Release/whisper.dll Binaries/
cp build/bin/Release/ggml.dll Binaries/
cp build/bin/Release/ggml-base.dll Binaries/
cp build/bin/Release/ggml-cpu.dll Binaries/
cp build/bin/Release/ggml-vulkan.dll Binaries/
```

Then download the GGML models (see [Models](#for-amd-gpu-ggml-format) above).

## Usage

```bash
python app.py
```

### File Transcription

1. Click **Select File** and choose an audio or video file
2. Set the output folder and options (model, format, device)
3. Click **Start Transcription**

The device dropdown auto-detects available GPUs. Select **Auto** to let the app choose the best option.

### Live Transcription

1. Switch to the **Live Transcription** tab
2. Select a loopback audio device (your speakers or headphones)
3. Set an output folder and click **Start Session**
4. Play audio (YouTube, Zoom, etc.) — words appear in real-time as streaming captions
5. Click **Stop** to end the session and save the transcript

Live transcription uses the Fast model on CPU for low-latency response, with 1-second audio overlap between buffers and context prompting for accuracy.

## Building an Executable

```bash
pip install pyinstaller
pyinstaller IvritTranscriber.spec
```

The executable will be in `dist/IvritTranscriber/`.

## Project Structure

```
app.py                      # Main application and GUI
core/
  settings.py               # Settings persistence (Pydantic)
  jobs.py                   # Job/Task state management
  worker.py                 # Transcription worker (QRunnable)
  live_worker.py            # Live transcription worker (QThread)
engine/
  audio_capture.py           # WASAPI loopback device enumeration and buffering
  ffmpeg_helper.py           # FFmpeg wrapper (probe, extract, split)
  model_loader.py            # CTranslate2 model validation and loading
  gpu_detector.py            # NVIDIA CUDA and AMD Vulkan detection
  transcriber.py             # Chunk transcription (faster-whisper)
  whisper_cpp_runner.py      # Chunk transcription (whisper.cpp subprocess)
  merger.py                  # Merge chunks into final SRT/TXT
  checkpoint.py              # Checkpoint support
ui/
  live_panel.py              # Live transcription UI panel
  settings_panel.py          # Settings UI panel (theme, model, device)
```

## License

MIT
