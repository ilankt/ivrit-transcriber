# Improvement Plan

This document outlines three improvements planned for Ivrit Transcriber:

1. **Faster startup** — splash screen + background initialization.
2. **Drop the Fast / Accurate model selector** — keep only the full `ivrit-large-v3` model.
3. **Language selection (English / Hebrew)** — with on-demand download of standard Whisper models for English.

No code changes have been made yet. This file is the implementation roadmap.

---

## 1. Faster Startup

### Problem

The app freezes for roughly 2-5 seconds before the window appears. The main thread is blocked inside `MainWindow.__init__()`:

| File:line | What it does | Cost |
|---|---|---|
| `app.py:132` | `detect_all_gpus()` — synchronous GPU detection | 1-3s |
| `engine/gpu_detector.py:23` | `import torch` (executed on first detection call) | major |
| `engine/gpu_detector.py:52`, `:72` | `vulkaninfo`, `wmic` subprocesses (10s timeout each) | up to 20s in pathological cases |
| `app.py:162` | `LiveTranscriptionPanel(self.settings)` — pulls in numpy + enumerates audio devices | 0.5-1.5s |

There is no splash screen. `window.show()` (`app.py:597`) only runs after all of the above completes, so the user sees nothing until everything is loaded.

### Approach: splash screen + background initialization

1. **Add a splash screen (`QSplashScreen`)** shown immediately after `QApplication` is constructed in the `__main__` block. Use `ICON.png` plus a status label that updates as background work progresses (e.g., "Detecting GPUs…", "Ready").

2. **Move GPU detection to a background `QThread`.** Add a `StartupWorker(QThread)` that runs `detect_all_gpus()` and emits a `gpu_info_ready` signal. The splash subscribes to status updates from this worker.

3. **Lazy-load the Live Transcription panel.** Replace the eager construction at `app.py:162` with a placeholder `QWidget`. Build `LiveTranscriptionPanel` only when its tab is first selected (connect to `QTabWidget.currentChanged`). This defers the numpy import and audio device enumeration until the user actually opens the tab.

4. **Construct `MainWindow` only after GPU detection completes.** When the startup worker emits `gpu_info_ready`, build the window with the GPU info ready, then close the splash.

5. **Lazy-import heavy modules.** `app.py` currently imports `from core.worker import TranscriptionWorker` at module load, which transitively imports `faster_whisper` (loads ctranslate2 C++ extension). Move this import inside the function that needs it (start-transcription handler). Same for `LiveTranscriptionPanel` — only import inside the lazy-loader.

### Files to modify

- `app.py` — add splash, add `StartupWorker`, refactor `__main__`, lazy-load Live Transcription panel, lazy-import `TranscriptionWorker`.

### Expected outcome

- Splash visible in <500ms (instant feedback).
- Main window visible in 1-2s instead of 2-5s.
- Live tab adds ~0.5-1.5s but only on first click, not at startup.

### Risks

- `closeEvent` at `app.py:575` calls `self.live_panel.stop_session()` — must handle the case where the panel was never built (check `hasattr` / `is not None`).
- Splash on Windows can flicker with high-DPI scaling — verify at 100% / 150% / 200%.

---

## 2. Drop "Fast / Accurate" Model Selector

### Problem

The current dual-model design adds UI complexity for a choice most users don't need:

| File:line | What it does |
|---|---|
| `core/settings.py:7` | `model_type: str = "Fast"` |
| `ui/settings_panel.py:46-48` | Model combo with `["Fast", "Accurate"]` |
| `core/worker.py` | Branches on `model_type` to pick `Models/ivrit-large-v3-turbo-ct2/` (beam_size=1) vs `Models/ivrit-large-v3-ct2/` (beam_size=3) |
| `engine/whisper_cpp_runner.py` (callers) | Branches on `model_type` to pick GGML file |
| `CLAUDE.md` | Documents the dual-model design |

### Approach: collapse to a single model per language, beam_size=3

1. **Remove `model_type` from Settings** (`core/settings.py:7`). Pydantic ignores unknown fields by default, so old `settings.json` files with `"model_type": "Fast"` will load cleanly.
2. **Remove the Model combo** from `ui/settings_panel.py` (combo at lines 46-48 plus its load/save lines at 105 and 125).
3. **Hardcode `beam_size = 3`** in `core/worker.py` and `core/live_worker.py` (replaces the previous branching).
4. **Remove turbo paths** from `core/worker.py` and `engine/whisper_cpp_runner.py` callers. The only Hebrew model is `ivrit-large-v3-ct2` (CT2) / `ggml-ivrit-large-v3.bin` (GGML).
5. **Manual cleanup step (documented, not auto-executed):** the user can delete `Models/ivrit-large-v3-turbo-ct2/` and any turbo `.bin` to reclaim disk space.
6. **Update `CLAUDE.md`** — drop Fast/Accurate from the Key Design Decisions section.

### Files to modify

- `core/settings.py`
- `ui/settings_panel.py`
- `core/worker.py`
- `core/live_worker.py`
- `engine/whisper_cpp_runner.py` (callers)
- `CLAUDE.md`

---

## 3. Language Selection (English / Hebrew)

### Problem

Language is hardcoded to Hebrew throughout:

| File:line | Hardcoded value |
|---|---|
| `core/worker.py:231` | `"he"` literal passed to `transcribe_chunk()` |
| `engine/whisper_cpp_runner.py:134` | `'--language', 'he',` in argv |
| `core/live_worker.py:332` | `language="he"` to `model.transcribe()` |
| `core/settings.py:6-17` | No `language` field |
| `ui/settings_panel.py:42-79` | No language combo |

The bundled ivrit models are Hebrew-only fine-tunes. They cannot transcribe English well — different models are needed for English.

### Approach: Settings + UI + per-language model resolution + on-demand download

#### 3.1 Settings layer

- Add `language: str = "he"` to `core/settings.py:Settings`. Default Hebrew preserves existing behavior on upgrade.

#### 3.2 UI layer

Add a `QComboBox` to `ui/settings_panel.py` in the Transcription group (replacing the now-removed Model combo's slot):

- `"Hebrew"` → `"he"` (default)
- `"English"` → `"en"`

Wire `_load_settings()` and `save_settings()` for the new combo. Add a small info label visible only when English is selected:

> *"English uses standard OpenAI Whisper. Model will be downloaded on first use (~3 GB)."*

#### 3.3 Model resolution

Add a resolver to `engine/model_loader.py`:

```python
def resolve_model_path(language: str, engine: str) -> str:
    """
    Returns the local path to the model directory or file.
    Triggers download if missing for English.
    Engines: 'faster-whisper' (CT2 dir), 'whisper-cpp' (GGML file).
    """
```

Mapping:

| language | engine          | local path                          | source repo (English only)                    |
|----------|-----------------|-------------------------------------|-----------------------------------------------|
| he       | faster-whisper  | `Models/ivrit-large-v3-ct2/`        | bundled                                       |
| he       | whisper-cpp     | `Models/ggml-ivrit-large-v3.bin`    | bundled                                       |
| en       | faster-whisper  | `Models/en-large-v3-ct2/`           | `Systran/faster-whisper-large-v3`             |
| en       | whisper-cpp     | `Models/ggml-large-v3.bin`          | `ggerganov/whisper.cpp` → `ggml-large-v3.bin` |

(Final HF repo IDs to be verified at implementation time.)

#### 3.4 On-demand download

Add `engine/model_downloader.py`:

- `download_ct2_model(repo_id, dest_dir, progress_cb)` — uses `huggingface_hub.snapshot_download()`.
- `download_ggml_file(repo_id, filename, dest_dir, progress_cb)` — uses `huggingface_hub.hf_hub_download()`.
- Validates results: directory exists, non-zero file sizes, CT2 model dir contains `model.bin`, `tokenizer.json`, `vocabulary.json`.

Download UX:

- When transcription starts and the model is missing, show a `QProgressDialog` with progress and a Cancel button. The download itself runs in a `QThread` so the UI stays responsive.
- Cancel → clean up partial download (`shutil.rmtree(dest_dir)`).
- Failure → `QMessageBox` error, abort job cleanly.

#### 3.5 Wire language through the pipeline

- `core/worker.py:231` — replace `"he"` with `self.settings.language`.
- `engine/whisper_cpp_runner.py` — add a `language` parameter to `transcribe_chunk_whispercpp()` (default `"he"`); use it at line 134 in place of hardcoded `'he'`. Pass `self.settings.language` from the worker.
- `core/live_worker.py:332` — replace `language="he"` with `language=self.settings.language`.
- `core/worker.py` model loading — call `resolve_model_path(self.settings.language, engine)`.
- `core/live_worker.py` model loading — same resolver.

### Files to modify

- `core/settings.py` — add `language` field.
- `ui/settings_panel.py` — add language combo, load/save, info label.
- `engine/model_loader.py` — add `resolve_model_path()`.
- `engine/model_downloader.py` — new file.
- `core/worker.py` — use resolver, pass language to both engines.
- `engine/whisper_cpp_runner.py` — accept `language` parameter.
- `core/live_worker.py` — use resolver, pass language.
- `app.py` — wire download progress dialog into start-transcription flow.
- `requirements.txt` — explicit `huggingface_hub` pin (it's already a transitive dep of faster-whisper, but pinning prevents surprises).

### Risks

- Existing disk space validation (H8) needs a higher threshold on first English use (~3 GB free).
- whisper.cpp GGML filename on the `ggerganov/whisper.cpp` HF repo must match what the engine expects — verify during implementation.

---

## Verification Plan

### Startup speed

- Add one-shot timing logs around `QApplication` creation, splash show, GPU detection complete, main window show. Remove before commit.
- Acceptance: main window visible in <2s on a typical dev machine.
- Click the Live tab — panel builds and audio device list populates on first click.
- Close the app while the Live tab was never opened — no exception (`closeEvent` handles `None` live panel).

### Model simplification

- Settings UI shows no Model combo. Only Language (plus theme/VAD/device/output format).
- Old `settings.json` containing `"model_type": "Fast"` loads cleanly; the field is silently dropped.
- Hebrew transcription on a known-good clip produces output equivalent to the previous "Accurate" mode (full v3, beam_size=3).
- Hebrew transcription on whisper.cpp engine still works (AMD path).

### Language — Hebrew

- Default install with no settings change → existing Hebrew transcription works.

### Language — English

- Fresh install with English selected → download dialog appears, model downloads, transcription proceeds.
- Re-run with English → no download, immediate start.
- Cancel mid-download → partial files cleaned up, no orphan directories under `Models/`.
- Network disconnect mid-download → graceful error message, no crash.
- Both engines (faster-whisper and whisper.cpp) work for English on appropriate hardware.

### Live transcription

- Switch language to English in Settings, start a live session → uses English model, produces English captions.

---

## Out of Scope / Future Work

- Translation (e.g., Hebrew → English) — ivrit models don't support it.
- Auto language detection — possible future addition; ivrit models have degraded detection so this would only be useful with the standard Whisper models.
- More languages (French, Spanish, etc.) — the resolver table is the only thing that needs to change to add more.
- Auto-deleting the old turbo model files — left as a manual cleanup step to avoid destructive automatic actions.
