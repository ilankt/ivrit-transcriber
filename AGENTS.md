# Repository Guidelines

## Project Structure & Module Organization

Ivrit Transcriber is a Python 3.11+ desktop app built with PySide6. The GUI entry point is `app.py`, which coordinates file selection, job creation, settings, and worker startup. Core state and orchestration live in `core/`: `settings.py` persists user preferences, `jobs.py` defines job/task state, `worker.py` handles file transcription, and `live_worker.py` handles live transcription. Transcription and media operations live in `engine/`, including FFmpeg helpers, GPU detection, model loading/downloading, Faster-Whisper, whisper.cpp, merging, and checkpoints. UI panels are in `ui/`. Runtime assets are local-only: `Models/` for Whisper models, `Binaries/` for whisper.cpp binaries, `logs/` for logs, and PyInstaller output in `build/` and `dist/`.

## Build, Test, and Development Commands

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the application locally with `python app.py`. Build the Windows executable with `pyinstaller IvritTranscriber.spec`; output is written under `dist/IvritTranscriber/`. FFmpeg must be available on `PATH`, and required models must exist under `Models/`.

## Coding Style & Naming Conventions

Use standard Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for Qt classes, and `UPPER_CASE` for constants. Keep modules focused on their current responsibility; avoid moving UI code into `engine/` or subprocess/media logic into `ui/`. Prefer explicit error handling around FFmpeg, model loading, filesystem cleanup, and Qt worker boundaries.

## Testing Guidelines

No automated test suite is currently tracked. Before submitting changes, run `python app.py` and smoke-test the touched workflow, such as file loading, settings persistence, live transcription startup, or output generation. When adding tests, use `pytest`, place them under `tests/`, and name files `test_<module>.py`.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, for example `Fix cleanup to skip .cache and other dot entries` and `Add manual model download and configurable models folder`. Follow that style and keep each commit scoped. Pull requests should describe the user-visible change, list manual verification steps, mention model/binary requirements when relevant, and include screenshots for UI changes.

## Security & Configuration Tips

Do not commit local models, whisper.cpp binaries, build artifacts, virtual environments, or logs. These are intentionally ignored in `.gitignore`. Keep generated transcripts and user media out of the repository unless they are small, sanitized fixtures.
