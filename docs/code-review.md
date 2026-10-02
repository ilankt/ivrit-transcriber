# Ivrit Transcriber code review

Reviewed on October 2, 2026. The review covered the application entry point,
all core, engine, and UI modules, tests, utility and benchmark scripts, dependency
files, launcher, and PyInstaller configuration. Concrete fixes are applied in the
working tree. No commit, push, or rebuilt executable was produced.

## Findings and fixes

Priority 1 findings risked losing or corrupting results; priority 2 findings
affected correctness, recovery, or responsiveness.

| Priority | Finding | Implemented change |
| --- | --- | --- |
| 1 | Closing the main window could destroy a running live or file-loading thread before results were saved. | Shutdown keeps processing Qt events until background work and live saving finish. A failed live save leaves the window open for recovery. |
| 1 | Stop set the same flag that made live inference discard its final segments, and final buffers of one second or less were skipped. | Finish the current inference, stop capture, and drain every nonempty final buffer. |
| 1 | Canceling or failing a model download could recursively delete a pre-existing model directory. | Preserve existing and partially downloaded files for retry. Keep the worker and dialog alive until native thread completion. |
| 1 | The File menu could replace the active job and delete audio still being used by transcription. | Guard the selection handler and disable both menu and button until the worker finishes. |
| 1 | Old checkpoints with the same output basename could be merged into a new transcript. | Assign a unique checkpoint namespace to every run; merge and remove only that run's checkpoints. Existing recovery files remain untouched. |
| 2 | Even a job where every chunk failed displayed success and deleted prepared audio. | Report partial or total failure as an error, retain prepared audio for retry, and reset task state on the next run. |
| 2 | Cancellation abandoned a daemon inference thread while reporting the job finished. | Keep inference owned by the job worker; observe cancellation at segment boundaries and raise instead of returning truncated text as a completed chunk. |
| 2 | whisper.cpp cancellation blocked indefinitely waiting for a stderr line. | Read stderr in a dedicated reader while polling cancellation; terminate, kill on timeout, and reap the subprocess before releasing temporary files. |
| 2 | Cleanup exceptions could prevent completion signals or falsely report deleted models. | Guard worker/audio teardown; propagate failed model-cleanup retries to the UI. Use native QThread completion signals separately from result signals. |
| 2 | Invalid settings could crash startup, and interrupted writes could truncate settings or transcripts. | Validate supported values and ranges, recover to defaults on read/validation errors, and atomically replace completed UTF-8 files. |
| 2 | Live sessions could start with stale controls or change language/export format midway through capture. | Synchronize controls before startup and snapshot settings in each worker. Export using the session's snapshot. |
| 2 | Failed live output writes left the panel stuck and put the transcript at risk. | Preserve unsaved segments, reset controls, and provide Save Session to retry in another folder. |
| 2 | Output directory errors escaped UI handlers; the fixed write-probe filename could overwrite an existing file. | Catch directory/permission failures and use a unique temporary file for the write probe. Log-file failure no longer prevents transcription. |
| 2 | CUDA detection required optional Torch, so a CPU-only Torch install could hide a usable Faster-Whisper GPU. | Detect CUDA using CTranslate2; optionally obtain the device name through nvidia-smi. Timed-out GPU probes now reap their child processes. |
| 2 | macOS input enumeration excluded common loopback drivers whose names did not contain “loopback.” | Enumerate available non-Windows input devices so users can select their configured loopback input. |
| 2 | SRT rounding could emit an invalid `00:00:60,000`; live timestamps rolled backward at midnight. | Format integer milliseconds with proper carry; export live timing relative to capture start. Display captions retain wall-clock timestamps. |
| 2 | Lexical chunk ordering placed chunk 1000 before chunk 101. | Sort prepared audio by numeric chunk index. |

Primary implementation files are [app.py](../app.py),
[the file worker](../core/worker.py), [the live worker](../core/live_worker.py),
[live UI](../ui/live_panel.py), [download UI](../ui/model_download.py),
[subprocess runner](../engine/whisper_cpp_runner.py),
[settings](../core/settings.py), and [atomic storage](../core/storage.py).

## Unused code and maintenance

- Removed the unused ETA signal/slot path and consolidated progress display on the stage-progress signal.
- Removed `get_whispercpp_binary_path` and `validate_whispercpp_binary`, which had no application callers; their tests now exercise the production resolver/error reporter.
- Removed the abandoned-thread cancellation helper, destructive download cleanup helper, unused downloader import, and the ignored whisper.cpp VAD argument.
- Labeled VAD's scope in Settings: live sessions and Faster-Whisper files. AMD/Metal file transcription does not apply this setting.
- Connected the previously inactive About action, replaced front-removal from the word list with a deque, and closed replaced logging handlers.
- Protected reserved Windows output stems; rejected empty/missing model files; handled empty audio buffers.
- Declared the Pydantic 2 requirement already used by the app, made optional speaker tests skip without their packages, ignored pytest cache files, and corrected the testing instructions and live-workflow documentation.
- Retained Qt callbacks and PyTorch adapter methods used dynamically, even when ordinary textual call counts would suggest they were unused.

## Verification

The original baseline was 136 passing tests. The first review reached 174 tests;
the follow-up adds recovery, queue, editor, bounded-audio, and model-repair tests in
[test_job_workflows.py](../tests/test_job_workflows.py). The final speaker-enabled
suite has 201 passing tests. A freshly created environment installed from the
core dependency snapshot passes 185 tests, with 3 optional speaker tests/modules
skipped because their packages are absent. `pip check` passes in that environment.
The checks cover corrupted settings, atomic-write failure, time rounding,
chunk ordering, checkpoint isolation, failed chunks, cooperative cancellation,
real silent-child cancellation, Qt shutdown, download lifetime, retained model
files, live flushing, and live-save recovery.

The app's source smoke check passed window construction/rendering, bundled voice
detection, and real M4A/MP4 decoding. All four tabs and the new transcript editor
were rendered with the native Windows Qt backend and visually inspected.
Screenshots and reports are local under `logs/workflow-*.png` and `logs/`.

Hardware checks used existing synthetic English speech: AMD/Vulkan and CPU
transcription both produced exports and durable completed records. DirectML
speaker inference passed the A/B/A/B fixture with two consistent speakers and no
GPU fallback. Windows WASAPI captured a generated two-second 440 Hz tone without
overflow; captured audio was inspected in memory and was not saved. The real
Faster-Whisper live pipeline also produced timed cues from an 11.1-second
synthetic fixture, exercising overlap, final-buffer draining, and export. Its
short final buffer included a spurious recognized word, so this check establishes
runtime/timestamp behavior, not transcription accuracy. No user media
was transcribed. UI smoke checks did not save user preferences.

## Follow-up improvements

- New jobs are saved atomically beside user settings, with completed chunk results,
  settings snapshots, source size/modification metadata, and speaker turns.
  Interrupted jobs recover as canceled and resume unfinished chunks. Output-write
  retries can re-export completed results without loading a model or needing audio.
- Jobs & Review supports multi-file queues, independent errors, restart recovery,
  and collision-free batch output names. Prepared audio survives failure/cancellation;
  success releases the owned cache. Manual cache release preserves transcript history.
- The transcript editor supports original-media playback, seeking, speed control,
  speaker renaming, text/timing corrections, and SRT/TXT exports. Corrections are
  separate from ASR results and survive completion of later chunks in a partial job.
- Live input is capped at 120 seconds or 64 MB, drained in ten-second batches, and
  reports backlog. Capture stops on overflow/driver discontinuity and drains accepted
  audio. Word timestamps reconcile overlap and create cues using speech timing.
- Speaker preprocessing reads ten seconds of PCM at a time into a disk-backed
  float32 waveform. Full-recording copies are eliminated; global speaker labeling
  remains intact. Cache preparation and mapped audio check free disk space.
- All four Hebrew/English backend combinations can be downloaded. The installed
  local models matched all published repository hashes during a read-only check.
  Files are pinned
  to one Hub revision, checked against SHA-256 or Git blob hashes, staged, and then
  replaced. Verify / Repair skips intact files. Interrupted CT2 updates are marked
  incomplete until repaired. Hebrew CT2 metadata comes from the
  [official ivrit.ai repository](https://huggingface.co/ivrit-ai/whisper-large-v3-ct2).
- Core and Windows speaker runtime snapshots pin dependencies. The core snapshot
  uses NumPy 2.4.1 because [2.4.1 fixes the withdrawn 2.4.0 compatibility issue](https://numpy.org/devdocs/release/2.4.1-notes.html).
  GitHub Actions is configured for Windows/macOS core tests and a Windows speaker
  suite, with offline model tests and uploaded smoke artifacts.

## Remaining limits

- NVIDIA and Apple hardware are unavailable on this machine. Their native inference remains unverified; CI configuration has been added but hosted runs require pushing these changes. Local routing tests are not hardware validation.
- No packaged EXE was rebuilt. The current spec excludes dependencies used by optional speaker detection; a speaker-enabled distribution needs its own dependency-inclusive build and packaged smoke validation.
- Live buffering is bounded, but throughput still depends on the model/device. A long-duration soak test and accuracy evaluation on varied speech remain useful. Short live buffers can still produce spurious words, as observed in the synthetic final-buffer check; overlap reconciliation does not solve model recognition errors. Live sessions are saved at Stop; restart recovery currently covers recorded-file jobs.
- Disk-backed speaker input reduces preprocessing RAM; model inference and clustering still allocate working memory. This is not a fixed bound on total process memory.
- Faster-Whisper cancellation waits for the current native inference boundary, file pause waits between chunks, and downloads wait for the current file. The app now reports and respects those lifecycle limits instead of abandoning running work.

The review does not imply that every possible defect has been excluded. The
remaining items above are documented rather than hidden by passing unit tests.
