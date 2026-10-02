# Ivrit Transcriber

A desktop application for transcribing Hebrew and English audio and video files on **Windows and macOS**. Built with Python, PySide6 (Qt), [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper), and [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Transcription runs locally and works offline once the required models are downloaded.

**Quick start:** [Windows](#installation) · [Apple Silicon Mac](#apple-silicon-mac-setup) · [Speaker labels (experimental)](#experimental-speaker-labels)

## Features

- Transcribe Hebrew audio and video files (MP3, WAV, MP4, MKV, etc.)
- **Live transcription** — capture system audio with streaming captions; Windows uses WASAPI loopback, while macOS requires an audio loopback input
- Outputs **SRT subtitles**, **plain text**, or both
- Standard Whisper large-v3 transcription model
- **GPU acceleration** — NVIDIA CUDA, AMD Vulkan, and Apple Silicon Metal (auto-detected)
- **Optional speaker labels** — local Speaker 1 / Speaker 2 labels, with word-level speaker changes in recorded-file exports (experimental)
- **Dark / Light / System theme** support
- Voice Activity Detection (VAD)
- Step-by-step progress, completion percentages, and estimated time remaining
- Custom output filenames

The latest source version includes GPU-assisted speaker detection and improved
two-speaker separation. On Windows, launch **Run Ivrit Transcriber.cmd**; use
`--setup` to install or repair its optional dependencies. Existing packaged
releases do not include these new speaker features. See the
[speaker setup guide](#experimental-speaker-labels) and
[validation notes](docs/speaker-recognition-testing.md) for requirements and known limitations.

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
[ivrit.ai Pyannote model](https://huggingface.co/ivrit-ai/pyannote-speaker-diarization-3.1)
pipeline detects speakers locally across the entire recording.

Run this experiment from source (existing packaged executables do not contain
the optional speaker dependencies):

```powershell
python -m pip install -r requirements.txt -r requirements-speakers.txt
python app.py
```

On Windows, **Run Ivrit Transcriber.cmd** opens the source app in its dedicated
`build/ivrit-speaker-env` environment (and installs it on first use using Python
3.12). Pass `--setup` to repair/install dependencies in an existing environment. Running
`python app.py` from another Python environment will not use those dependencies.

In **Settings > Set Up Speakers**, download the public model files. No access
token is normally required. An optional token is used only for the download and
is not saved by the app. The models are stored in the selected Models folder;
after setup, detection works offline. Pyannote usage telemetry is disabled.

Enable **Detect speakers (recorded files only)**. Leave **Speakers** on Auto or
choose the known count. Detection runs before transcription and adds processing
time and memory use. Speaker acceleration follows the selected device:

| Selected device | Speaker processing | Validation |
| --- | --- | --- |
| NVIDIA GPU | PyTorch CUDA for segmentation and embeddings | Device routing and CPU recovery tested; native NVIDIA run still needed |
| AMD GPU on Windows | DirectML for speech filters and the embedding encoder; CPU for recurrent tracking and clustering | Inference and full transcription/export tested on RX 7600M XT |
| Apple GPU (Metal) | PyTorch MPS for speech filters and the embedding encoder; CPU for recurrent tracking and clustering | Device routing, adapter math, and recovery tested; native Apple Silicon run still needed |
| CPU Only | CPU for all stages | Tested |

Auto prefers CUDA on Windows/Linux, MPS on a compatible Mac, and DirectML on
Windows without CUDA. CPU Only never initializes a speaker GPU. A Mac needs
native ARM64 Python and an MPS-compatible macOS/PyTorch installation for Metal;
the current optional PyTorch 2.7.1 package set does not support Intel macOS wheels.

The Windows launcher chooses accelerator dependencies during first setup or
`--setup`: CUDA when `nvidia-smi` detects NVIDIA, otherwise DirectML. For manual
setup, run this **after** installing the regular and speaker dependencies:

```powershell
python scripts/setup_speaker_acceleration.py
```

NVIDIA setup installs matching CUDA-enabled Torch and TorchAudio wheels from
PyTorch's official index, replacing a CPU-only installation. CUDA 12.8 is the
default; `--backend cuda --cuda-version cu126` or `cu118` can be used for a
compatible older driver/GPU combination. Drivers must already support the
selected CUDA build. On Apple Silicon no extra accelerator package is required;
Metal is included in native PyTorch. See [PyTorch installation options](https://pytorch.org/get-started/previous-versions/#v271)
and [MPS requirements](https://docs.pytorch.org/docs/2.7/notes/mps.html).

To explicitly install the Windows DirectML path:

```powershell
python -m pip install -r requirements-speakers-amd.txt
python -m pip install --force-reinstall --no-deps onnxruntime-directml==1.24.4
```

The final reinstall prevents the CPU ONNX package required by Faster-Whisper
from overwriting DirectML's shared runtime files. The first DirectML run creates
local derived ONNX models beside the downloaded weights; subsequent runs reuse
them offline. Finding speech shows **GPU + CPU** on AMD/Metal because the filters
use the GPU while recurrent tracking stays on CPU. Near-silent windows also use
CPU to preserve numerical accuracy. Progress shows which stages use GPU or CPU. If GPU initialization
or execution fails, processing continues on CPU and the status reports the
fallback. DirectML and Metal retry a failed filter/embedding batch on CPU; a CUDA failure
restarts speaker analysis on CPU. Cancellation is preserved during recovery.
DirectML currently uses the default Windows graphics adapter; multi-adapter
selection still needs validation.

Beneath the status, **Step 1/3**, **Step 2/3**, and **Step 3/3** indicate finding
speech, comparing voices, and transcription. The progress bar and percentage
refer to the current step. Time remaining is estimated from that step's measured
throughput, excludes pauses, and initially shows **Estimating time remaining**.
With speaker detection disabled, transcription is **Step 1/1**.

Speaker numbers are assigned in order of first appearance and are consistent
within a recording, not across recordings. Both Faster-Whisper and whisper.cpp
use word timestamps to split text at speaker changes. If word timings are
missing or incomplete, a subtitle keeps its full text and dominant speaker.
Selecting **2 speakers** also keeps locally distinct voices from both choosing
the same speaker label during voice matching. This improves separation but
does not guarantee accuracy: overlapping voices, timing errors, and similar
voices can still produce incorrect labels.
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
