# Ivrit Transcriber

A desktop application for transcribing Hebrew and English audio and video files on **Windows and macOS**. Built with Python, PySide6 (Qt), [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper), and [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Transcription runs locally and works offline once the required models are downloaded.

**Quick start:** [Windows](#installation) · [Apple Silicon Mac](#apple-silicon-mac-setup) · [Speaker labels (experimental)](#experimental-speaker-labels)

## Features

- Transcribe Hebrew audio and video files (MP3, WAV, MP4, MKV, etc.)
- **Live transcription** — capture system audio with streaming captions; Windows uses WASAPI loopback, while macOS requires an audio loopback input
- Outputs **SRT subtitles**, **plain text**, or both
- Standard Whisper large-v3 transcription model
- **GPU acceleration** — NVIDIA CUDA, AMD Vulkan, and Apple Silicon Metal (auto-detected)
- **Dark / Light / System theme** support
- Voice Activity Detection (VAD)
- Progress tracking with ETA
- Custom output filenames

## Platform support

| Platform | File transcription | Live transcription |
| --- | --- | --- |
| Windows | CPU, NVIDIA CUDA, or AMD Vulkan | System audio via WASAPI loopback |
| Apple Silicon Mac | Metal GPU via whisper.cpp; CPU also available | CPU; requires a configured audio loopback input |
| Intel Mac | CPU via Faster-Whisper | CPU; requires a configured audio loopback input |

Mac users run the app from source using the setup below. The Windows executable
does not run on macOS, and this repository does not currently provide a packaged
Mac app. Metal acceleration applies to file transcription, not live transcription.

## Experimental speaker labels

Recorded files can optionally include **Speaker 1**, **Speaker 2**, etc. in TXT
and SRT exports. The existing transcription model is unchanged; a separate
[Pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
pipeline detects speakers locally across the entire recording.

Run this experiment from source (existing packaged executables do not contain
the optional speaker dependencies):

```powershell
python -m pip install -r requirements.txt -r requirements-speakers.txt
python app.py
```

In **Settings > Set Up Speakers**, follow the link to accept the model's Hugging
Face access conditions, provide a read token, and download the model. The token
is used only for the download and is not saved by the app. An existing Hugging
Face login can also be used. The model is stored in the selected Models folder;
after setup, detection works offline. Pyannote usage telemetry is disabled.

Enable **Detect speakers (recorded files only)**. Leave **Speakers** on Auto or
choose the known count. Detection runs before transcription and adds processing
time and memory use. NVIDIA CUDA is used when available and selected; CPU is used
otherwise, including when transcription uses AMD Vulkan or Apple Metal.

Speaker numbers are assigned in order of first appearance and are consistent
within a recording, not across recordings. Faster-Whisper uses word timestamps
to split text at speaker changes. With whisper.cpp, each subtitle segment gets
its dominant speaker; rapid exchanges inside a segment may be attributed
incorrectly. Overlapping voices and similar voices can also produce errors.
Text without an overlapping detected speaker is marked **Speaker unknown**.
Live transcription does not use this feature.

Cancel/pause during detection takes effect at the next pipeline progress hook;
during model download, cancellation waits for the current file to finish.
Turning speaker detection off uses the regular transcription workflow.

## Requirements

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available in PATH
- Hebrew Whisper models (see [Models](#models) below)
- Optional: PyTorch with CUDA for NVIDIA GPU acceleration
- Optional: whisper.cpp with Vulkan for AMD GPU acceleration (see [AMD GPU Setup](#amd-gpu-setup))

## Installation

Windows source installation (Mac users should follow [Mac setup](#apple-silicon-mac-setup)):

```powershell
git clone https://github.com/ilankt/ivrit-transcriber.git
cd ivrit-transcriber
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Models

The app uses the standard large-v3 Whisper model for transcription. Which format you need depends on your setup:

### For CPU or NVIDIA GPU (CTranslate2 format)

Download and place in `Models/`:

```
Models/
  ivrit-large-v3-ct2/
```

Each folder must contain: `model.bin`, `tokenizer.json`, `vocabulary.json`.

### For AMD GPU (GGML format)

Apple Silicon Metal uses this format too; the Mac setup below can download it
through Settings.

Download and place in `Models/`:

```bash
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

## Apple Silicon Mac setup

File transcription on Apple Silicon uses whisper.cpp with Metal when **Auto**
or **Apple GPU (Metal)** is selected. **CPU Only** continues to use Faster-Whisper.
Intel Macs use the CPU path. Apple Silicon is also detected when Python runs
under Rosetta, but native ARM64 Python and Homebrew are recommended.

Install the [Homebrew whisper.cpp package](https://formulae.brew.sh/formula/whisper.cpp)
and FFmpeg, then set up the app from this repository:

```bash
brew install whisper.cpp ffmpeg python@3.12
git clone https://github.com/ilankt/ivrit-transcriber.git
cd ivrit-transcriber
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

In Settings, select **Auto** or **Apple GPU (Metal)** and download the model if
needed. Hebrew uses `ggml-ivrit-large-v3.bin` (the app renames the downloaded
`ggml-model.bin`); English uses `ggml-large-v3.bin`. Each download is about 3 GB.
Existing CTranslate2 model folders are used only by the CPU/NVIDIA backend.
If your GGML models are already elsewhere, select that folder in Settings.

The app finds `whisper-cli`, `ffmpeg`, and `ffprobe` in PATH and the standard
Homebrew locations, including when launched from Finder. A missing or broken
whisper.cpp installation produces a setup error instead of silently selecting
Faster-Whisper on the CPU. Logs include whisper.cpp's GPU backend initialization.
Metal inference is provided by [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

This acceleration applies to file transcription. Live transcription still uses
Faster-Whisper on the CPU on Macs; system-audio capture requires an input device
provided by an audio loopback driver.

For an Intel Mac, install FFmpeg and Python, use the same source installation
steps, and select **CPU Only**. Supply the Hebrew CTranslate2 model described
above, or select English and download its model in Settings.

## Usage

```bash
python app.py
```

### File Transcription

1. Click **Select File** and choose an audio or video file
2. Set the output folder and options (language, format, device)
3. Click **Start Transcription**

The device dropdown auto-detects available GPUs. Select **Auto** to let the app choose the best option.

### Live Transcription

1. Switch to the **Live Transcription** tab
2. Select a loopback audio device (your speakers or headphones)
3. Set an output folder and click **Start Session**
4. Play audio (YouTube, Zoom, etc.) — words appear in real-time as streaming captions
5. Click **Stop** to end the session and save the transcript

Live transcription uses faster-whisper on CPU for low-latency response, with 1-second audio overlap between buffers and context prompting for accuracy.

## Building an Executable

```bash
pip install pyinstaller
pyinstaller IvritTranscriber.spec
```

The executable will be in `dist/IvritTranscriber/`.

Verify the packaged runtime before installing it:

```powershell
python scripts/smoke_test_exe.py dist/IvritTranscriber/IvritTranscriber.exe
```

This check starts the actual EXE without showing a window, renders its main
window, runs bundled voice detection, and decodes generated MP4/M4A samples.
It writes a JSON report and window image under `logs/`, exits automatically,
and does not save application settings. A process that merely stays running
does not count as a successful startup check.

## Project Structure

```
app.py                      # Main application and GUI
core/
  filenames.py              # Shared output filename sanitizing
  settings.py               # Settings persistence (Pydantic)
  jobs.py                   # Job/Task state dataclasses
  runtime.py                # Runtime path and engine selection helpers
  worker.py                 # Transcription worker (QRunnable)
  live_worker.py            # Live transcription worker (QThread)
engine/
  audio_capture.py           # WASAPI loopback device enumeration and buffering
  checkpoint.py              # Progressive save and final SRT/TXT merge support
  ffmpeg_helper.py           # FFmpeg wrapper (probe, extract, split)
  model_loader.py            # Model registry, validation, downloads, and loading
  gpu_detector.py            # NVIDIA CUDA, AMD Vulkan, and Apple Metal detection
  transcriber.py             # Chunk transcription (faster-whisper)
  whisper_cpp_runner.py      # Chunk transcription (whisper.cpp subprocess)
ui/
  live_panel.py              # Live transcription UI panel
  settings_panel.py          # Settings UI panel (theme, model, device)
```

## License

MIT
