# Repository Guidelines

## Project Structure & Module Organization

Ivrit Transcriber is a Python 3.11+ desktop app built with PySide6. The GUI entry point is `app.py`, which coordinates file selection, job creation, settings, and worker startup. Core state and orchestration live in `core/`: `settings.py` persists user preferences, `jobs.py` defines job/task state, `runtime.py` holds shared runtime helpers, `worker.py` handles file transcription, and `live_worker.py` handles live transcription. Transcription and media operations live in `engine/`, including FFmpeg helpers, GPU detection, model loading/downloading, Faster-Whisper, whisper.cpp, and checkpoint-based output merging. UI panels are in `ui/`. Runtime assets are local-only: `Models/` for Whisper models, `Binaries/` for whisper.cpp binaries, `logs/` for logs, and PyInstaller output in `build/` and `dist/`.

## Build, Test, and Development Commands

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the application locally with `python app.py`. Build the Windows executable with `pyinstaller IvritTranscriber.spec`; output is written under `dist/IvritTranscriber/`. FFmpeg must be available on `PATH`, and required models must exist under `Models/`.

For reproducible source development, install `requirements-lock.txt` and `requirements-test.txt`. The separate `requirements-speakers-lock.txt` captures the tested Windows/Python 3.12 speaker runtime; do not combine these two runtime snapshots. The Windows launcher uses the speaker snapshot. Keep job records, cached audio, corrections, and hardware test output out of Git.

## Coding Style & Naming Conventions

Use standard Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for Qt classes, and `UPPER_CASE` for constants. Keep modules focused on their current responsibility; avoid moving UI code into `engine/` or subprocess/media logic into `ui/`. Prefer explicit error handling around FFmpeg, model loading, filesystem cleanup, and Qt worker boundaries.

## Testing Guidelines

Run `python -m pytest -q` for the automated suite in `tests/`. Tests for optional speaker adapters skip when their dependencies are absent; use the speaker-enabled environment for full coverage. Before submitting changes, run `python app.py --smoke-test-report logs/source-smoke-report.json` to render the GUI and exercise VAD and media decoding without saving settings. Also smoke-test touched hardware workflows when the required models and devices are available. Name new pytest files `test_<module>.py`.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, for example `Fix cleanup to skip .cache and other dot entries` and `Add manual model download and configurable models folder`. Follow that style and keep each commit scoped. Pull requests should describe the user-visible change, list manual verification steps, mention model/binary requirements when relevant, and include screenshots for UI changes.

## Security & Configuration Tips

Do not commit local models, whisper.cpp binaries, build artifacts, virtual environments, or logs. These are intentionally ignored in `.gitignore`. Keep generated transcripts and user media out of the repository unless they are small, sanitized fixtures.
