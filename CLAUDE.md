# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ivrit Transcriber is a desktop application for transcribing Hebrew audio and video files. It supports two transcription backends: Faster-Whisper (CTranslate2, for NVIDIA/CPU) and whisper.cpp (Vulkan, for AMD GPUs). Built with Python 3.11+ and PySide6 (Qt), providing an offline transcription solution with a GUI.

## Architecture

### Core Components

**Application Entry Point:**
- `app.py` - Main PySide6 application with `MainWindow` class that coordinates the UI and transcription workflow

**Processing Pipeline:**
1. File input → Media probing (FFmpeg)
2. Video files → Audio extraction (FFmpeg)
3. Audio splitting into 1-minute chunks (FFmpeg)
4. Chunk-by-chunk transcription (Faster-Whisper or whisper.cpp depending on device)
5. Merging outputs to `.txt` and `.srt` files

**Module Structure:**
- `core/settings.py` - Settings persistence using Pydantic models (saved to `%APPDATA%/IvritTranscriber/settings.json` on Windows, `~/.ivrit_transcriber/settings.json` on Linux)
- `core/jobs.py` - Job and Task state management with enums for status tracking
- `core/worker.py` - `TranscriptionWorker` QRunnable that processes jobs asynchronously using Qt's thread pool; routes to the correct engine based on device setting
- `engine/ffmpeg_helper.py` - FFmpeg wrapper functions for probing, audio extraction, and splitting
- `engine/model_loader.py` - CTranslate2 model validation and Faster-Whisper initialization with GPU support
- `engine/gpu_detector.py` - GPU detection for both NVIDIA CUDA and AMD Vulkan
- `engine/transcriber.py` - Chunk transcription logic using Faster-Whisper (NVIDIA/CPU path)
- `engine/whisper_cpp_runner.py` - Chunk transcription via whisper.cpp subprocess (AMD GPU path)
- `engine/merger.py` - Merges transcribed chunks into final `.txt` and `.srt` files with proper time offsets (uses JSON format for robustness)

### Key Design Decisions

- **Sequential Processing**: Jobs are processed one at a time (`QThreadPool.setMaxThreadCount(1)`)
- **Temporary Storage**: Each job creates a temp directory for intermediate files (audio extraction, chunks)
- **Chunk Duration**: Audio is split into 1-minute chunks (hardcoded in `app.py`)
- **Model Selection**: "Fast" uses `ivrit-large-v3-turbo-ct2` with beam_size=1, "Accurate" uses `ivrit-large-v3-ct2` with beam_size=3
- **Model Location**: Models are expected in `Models/` subdirectory relative to the application (supports PyInstaller packaging via `get_base_path()` in `worker.py`)
- **Device Selection**: User can choose Auto/CPU/NVIDIA GPU/AMD GPU via UI. GPU options only shown if compatible hardware is detected. Device setting saved in settings.json.
- **Dual Engine**: NVIDIA/CPU uses faster-whisper (CTranslate2 models). AMD uses whisper.cpp with Vulkan (GGML models). Engine is auto-selected based on device setting.
- **whisper.cpp Integration**: Called as subprocess (like FFmpeg). Binary expected in `Binaries/` or system PATH. GGML models expected in `Models/`.

## Development Commands

### Running the Application
```bash
python app.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```
Dependencies: PySide6, faster-whisper, ffmpeg-python, tqdm, pydantic

Optional: PyTorch with CUDA support for GPU acceleration (if not installed, GPU option will not be available)

### Building Executable
```bash
pyinstaller IvritTranscriber.spec
```
Note: Requires FFmpeg binaries available in system PATH or bundled with the application.

## Working with the Codebase

### Job Processing Flow
1. `MainWindow._add_files()` handles file selection, probes media, extracts audio (if video), splits into chunks, creates `Job` and `Task` objects
2. `MainWindow._start_transcription()` creates `TranscriptionWorker` instances for each job and adds them to the thread pool
3. `TranscriptionWorker.run()` loads the model, transcribes each task (chunk), emits progress/ETA signals
4. After all tasks complete, merger functions combine outputs and save to output directory
5. Temporary directories are cleaned up on job completion or app exit

### Signal Flow
The worker emits Qt signals to update the UI:
- `job_status_updated` - Job status changes (QUEUED, RUNNING, PAUSED, DONE, ERROR, CANCELED)
- `task_status_updated` - Individual chunk status with messages
- `progress_updated` - Per-task progress (0.0 to 1.0)
- `eta_updated` - Estimated time remaining based on throughput
- `finished` - Worker cleanup trigger

### Settings Management
Settings are loaded at startup and saved on app exit. Configurable settings include:
- **Model Type**: "Fast" (turbo model, beam_size=1) or "Accurate" (full model, beam_size=3)
- **Device**: "Auto" (try GPU, fallback to CPU), "CPU Only", or "GPU Only" (only shown if NVIDIA CUDA detected)
- **VAD**: Voice Activity Detection can be toggled on/off
- **Output Folder**: Last used output folder is remembered
- **Advanced** (in settings file only): compute_type, threads

### SRT Time Offset Calculation
The merger accumulates time offsets based on the end time of the last segment in each chunk, ensuring continuous timeline across merged subtitles (`merger.py:36-39`).

## Known Issues

- Cancellation is not immediate due to blocking FFmpeg and Faster-Whisper operations (see `Gemini.md:111-113`)
- FFmpeg wizard for first-run setup is not yet implemented (see `Gemini.md:80-81`)

## Recent Improvements (Phase 1 & Phase 2)

### Phase 1: Critical Fixes (Completed)
- ✅ Fixed SRT segment parsing bug using JSON format instead of comma-separated values (`merger.py`)
- ✅ Fixed race condition in worker cleanup dictionary iteration (`app.py:350`)
- ✅ Added error handling for FFmpeg operations with proper cleanup (`app.py`)
- ✅ Fixed unsafe array access in merger with try-except blocks (`merger.py`)
- ✅ Centralized temp directory cleanup in worker's finally block
- ✅ Added missing `__init__.py` files for proper package structure
- ✅ Fixed FFmpeg error message decoding (`ffmpeg_helper.py`)
- ✅ Added model path validation before loading (`worker.py:85-91`)
- ✅ Fixed hardcoded model paths for PyInstaller support (`worker.py:get_base_path()`)

### Phase 2: High-Priority Features (Partially Completed)
- ✅ **GPU Support (FR1)**: Added device selection UI with NVIDIA CUDA detection
  - Device dropdown in Options pane
  - GPU option only shown if compatible NVIDIA GPU detected
  - Settings persisted across sessions
  - Files: `engine/gpu_detector.py`, `core/settings.py`, `app.py`, `worker.py`
- ✅ **Disk Space Validation (H7)**: Check available disk space before starting transcription
- ✅ **Output Directory Validation (H8)**: Verify write permissions before starting
- ✅ **Open Output Folder Button (L2)**: Added button to open output folder in system file explorer
  - Cross-platform support (Windows, macOS, Linux)

## Model Files

Models must be CTranslate2 format with required files:
- `model.bin`
- `tokenizer.json`
- `vocabulary.json`

Validated by `model_loader.validate_model_path()`.
