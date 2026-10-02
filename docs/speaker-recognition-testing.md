# Speaker recognition experiment

Started on 2026-10-02 on `codex/speaker-recognition-testing`, after merging and
pushing the existing work and Mac documentation to `main` (`58a9203`).

## First results

These are local Windows CPU integration checks, using a 50.235-second synthetic
English recording with Microsoft David and Zira voices alternating A/B/A/B.
The test supplies known sentence intervals and verifies that the same person
receives the same label when they return, and that A and B receive different
labels. It does not test ASR, word timing, overlap, or Hebrew accuracy.

| Configuration | Result |
| --- | --- |
| App's Community-1 model | Download blocked by `GatedRepoError` with the existing Hugging Face login |
| ivrit 0.2.6, Pyannote 3.3.2, unmodified wrapper | Model assets downloaded and inference ran, then label assignment failed with `NameError` |
| Same ivrit stack, test-only NumPy import workaround, fixed count of 2 | Passed A/B/A/B consistency; 19.44 seconds including loading |
| Same workaround, automatic speaker count | Passed A/B/A/B consistency; 2 assigned labels; 19.36 seconds including loading |

The installed ivrit wrapper's `_match_speaker_to_interval` uses `np.minimum` and
`np.maximum`, but NumPy is imported only under `TYPE_CHECKING` and within another
method. `--ivrit-numpy-workaround` supplies `ivrit.diarization.np` in the test
process. It does not edit the installed package or change the application's
backend. The wrapper also needed `matplotlib`, which its resolved dependencies
did not install automatically.

The tested ivrit stack uses `ivrit-ai/pyannote-speaker-diarization-3.1`, its
segmentation model, and the Pyannote WeSpeaker embedding model. The published
assets downloaded successfully using the existing environment. This does not
establish that every user's account/setup can download them.

At the time of these first tests the app still used Community-1. The subsequent
integration below changes that on this testing branch.

## Source app integration

The source app now uses the same ivrit.ai Pyannote 3.1 model directly through
Pyannote 3.3.2. It loads local segmentation and embedding files and uses the
app's existing speaker-to-word assignment, avoiding the ivrit wrapper's missing
NumPy import entirely. Downloads are anonymous by default and stored in the
configured Models folder. Community-1 is no longer required by this branch.

Use **Run Ivrit Transcriber.cmd** from the project root on Windows. It selects
`build/ivrit-speaker-env` explicitly, so a system Python or a different active
virtual environment cannot accidentally launch the app without speaker support.
The initial setup requires Python 3.12. Alternatively, activate that environment
before running `python app.py`.

The integrated app backend passed the synthetic A/B/A/B speaker check with
automatic counting, with actual model-detected intervals rather than the supplied
reference intervals. Re-run this check with `--backend app`.

The complete source-app worker also passed an offline run of that sample using
the local English Whisper model on CPU: media loading, speaker inference,
word-timed transcription, and both TXT/SRT exports containing Speaker 1 and
Speaker 2. This run exposed a Faster-Whisper 1.2.1 incompatibility with PyAV 19;
the app now constrains PyAV below 19 and includes a real decoder regression test.
At this point the automated suite passed 66 tests. This still does not validate accuracy
on real Hebrew conversations.

## AMD GPU acceleration

The Windows AMD path now exports only WeSpeaker ResNet34's frame encoder to
ONNX Runtime DirectML. It keeps the original weights, CPU filterbank extraction,
weighted pooling (including overlap masks), segmentation, and clustering. No
replacement diarization model, cloud service, access token, or driver change is
required. The derived ONNX file is cached in the speaker model folder and its
filename includes a hash of the source weights.

Measured on the local Radeon RX 7600M XT, Windows, PyTorch 2.7.1,
Pyannote 3.3.2, ONNX Runtime DirectML 1.24.4:

| Synthetic recording | CPU speaker inference | DirectML speaker inference | Result |
| --- | --- | --- | --- |
| 50.235 seconds, initial prototype, 8 CPU threads | 17.26 s | 1.44 s | Identical labels and intervals |
| 200.94 seconds, integrated adapter, default 16 CPU threads | 55.47 s | 5.30 s | Identical labels and intervals across all four repetitions |

These are inference-only timings on synthetic English voices, not real Hebrew
accuracy or end-to-end transcription benchmarks. The longer run separately
measured 4.91 seconds for model loading and 0.70 seconds for GPU setup/export.
GPU inference follows numerical checks that warm the runtime; cold runs can
take longer. The longer fixture repeats the original A/B/A/B sample four times.
The encoder's runtime profile listed only `DmlExecutionProvider` for executed
nodes. Numerical checks also passed for batches 1/3/8, 1/2/10-second windows,
unmasked inputs, random speech masks, and zero-weight masks.

The application's real worker completed an offline test with `device="amd"`,
DirectML speaker detection, whisper.cpp transcription, and TXT/SRT exports with
both labels. The GUI/VAD/media smoke check passed after installing DirectML.
At this stage 73 automated tests passed, including unavailable-GPU setup, runtime failure,
nonfinite GPU outputs, CPU retry, and progress reporting.

AMD (or Auto without CUDA) enables DirectML when installed. CPU selection never
initializes it. Progress distinguishes CPU speech segmentation from GPU voice
comparison; a GPU failure retries the affected batch on CPU and reports that
fallback. The adapter currently uses the default DirectX device, validated on
this machine's single reported GPU. Multiple-adapter selection and other AMD
models still require testing. First-use export and session setup happen before
the next cancellation checkpoint, so cancellation can briefly wait for setup.

To reproduce after the normal speaker setup:

```powershell
python -m pip install -r requirements-speakers-amd.txt
python -m pip install --force-reinstall --no-deps onnxruntime-directml==1.24.4
python scripts/benchmark_speaker_directml.py --repeat 4
```

The final install ensures DirectML's files take precedence over the CPU ONNX
runtime installed by Faster-Whisper. Reports and execution profiles are local
under `build/speaker-directml/`. The source launcher includes these installation
steps when invoked with `--setup` on a Windows host without NVIDIA.

## NVIDIA and Apple Silicon paths

Device policy is isolated in `engine/speaker_devices.py` and honors explicit CPU
selection. NVIDIA retains full Pyannote CUDA inference. If CUDA initialization
partially moves a pipeline or execution fails, a fresh CPU pipeline is loaded
and analysis is retried once. User cancellation does not trigger a CPU retry.

Apple Silicon uses `engine/speaker_metal.py` to put a copy of the ResNet frame
encoder on MPS. Audio FFT/filterbanks, weighted pooling, segmentation, and
clustering remain on CPU. Keeping that boundary avoids dependence on MPS
support for audio FFT/vmap or recurrent segmentation layers. The original CPU
model remains available if a GPU operation fails or returns nonfinite output;
only that batch and subsequent batches use the CPU fallback.

`scripts/setup_speaker_acceleration.py` chooses CUDA packages for an NVIDIA host,
DirectML for other Windows hosts, and the already-bundled MPS backend for native
Apple Silicon Python. `--dry-run` prints installation commands without changing
anything; explicit backend and CUDA-build overrides are supported. The Windows
launcher invokes this during dependency setup. AMD-specific ONNX packages are
not needed by the CUDA or MPS paths.

The current suite passes **105 tests**: Windows/macOS/Linux selection, explicit
CPU behavior, CUDA setup/execution recovery, cancellation, MPS adapter math on a
CPU test device, MPS failure recovery, and the existing application tests. The
strict actual AMD hardware check also passed after these changes (7.39 seconds
including speaker loading and setup for the 50.235-second sample, no fallback).
There is **no native NVIDIA or Mac hardware validation from this Windows AMD
machine**. Passing policy tests is not proof of native device compatibility.

For native validation, copy the generated synthetic `build/speaker-sample`
directory (including its four source WAVs and `parts.json`) to the test machine,
install the source app's dependencies and download its speaker models, then run
the appropriate command:

```text
python scripts/test_speaker_models.py --backend app --device nvidia --require-gpu
python scripts/test_speaker_models.py --backend app --device metal --require-gpu
python scripts/test_speaker_models.py --backend app --device amd --require-gpu
```

`--sample-dir` accepts another location for that fixture. The hardware check
requires correct returning-speaker assignments and GPU use without fallback;
a CPU-only run cannot silently pass. CUDA/Metal speed and output quality still
need native measurements on representative real recordings before a release.

## Repeat the checks

Use a Pyannote 3.x environment for the app and ivrit experiment; retain a separate
Pyannote 4.x environment only if comparing Community-1:

```powershell
python -m venv build/ivrit-speaker-env
build/ivrit-speaker-env/Scripts/python.exe -m pip install -r scripts/requirements-speaker-testing.txt
./scripts/make_speaker_sample.ps1
build/ivrit-speaker-env/Scripts/python.exe scripts/test_speaker_models.py --backend ivrit --speakers 2
build/ivrit-speaker-env/Scripts/python.exe scripts/test_speaker_models.py --backend ivrit --speakers 2 --ivrit-numpy-workaround
build/ivrit-speaker-env/Scripts/python.exe scripts/test_speaker_models.py --backend ivrit --ivrit-numpy-workaround
```

The first inference command reproduces the upstream wrapper error. The next two
apply the explicit workaround. The generator requires the Windows desktop David
and Zira voices and produces no user-derived audio.

In a separate environment with Pyannote 4.x installed, compare Community-1:

```powershell
python scripts/test_speaker_models.py --backend community --speakers 2
```

Reports and generated audio remain under the ignored `build/speaker-sample/`
directory. Reports contain package versions, timings, speaker assignments and
exception types; they exclude credentials and transcript text. ivrit's reported
intervals are the supplied reference intervals, not measured speaker boundaries.
The Community-1 report contains the model's detected intervals instead.

## Public conversation accuracy check

`scripts/benchmark_speaker_clustering.py` uses only fixed public fixtures from
the [Pyannote 3.3.2 repository](https://github.com/pyannote/pyannote-audio/tree/3.3.2).
It does not read application settings, recent files, or user recordings.
Place these downloads under the ignored `build/public-speaker-sample/` directory:

- `pyannote/audio/sample/sample.wav` and `sample.rttm`.
- `tests/data/dev00.wav` and `dev01.wav`.
- `tests/data/debug.development.rttm`, renamed to `development.rttm`.
- `tests/data/trn03.wav` and `debug.train.rttm`, renamed to `training.rttm`.
- `tests/data/tst00.wav`, `tst01.wav`, and `debug.test.rttm`.

With the speaker model and DirectML installed, run:

```powershell
build/ivrit-speaker-env/Scripts/python.exe scripts/benchmark_speaker_clustering.py
```

The report is `build/public-speaker-sample/clustering-comparison.json`.
The sample, dev00, dev01, and trn03 clips have two reference speakers; tst00
and tst01 have four reference identities. Evaluation covers the first
30 seconds, includes overlap, and uses zero boundary tolerance. The diarization
error rate combines speaker confusion, missed speech, and false speech detection;
it is not a transcript word error rate.

| Public fixture | Original assignment | Joint assignment, original clustering thresholds |
| --- | ---: | ---: |
| sample | 7.43% | 5.70% |
| dev00 | 35.15% | 34.98% |
| dev01 | 52.35% | 49.95% |
| trn03 | 39.68% | 30.21% |
| tst00 (4 speakers) | 46.41% | 49.03% |
| tst01 (4 speakers) | 52.72% | 52.72% |

Variants reuse identical segmentations and embeddings so the comparison isolates
clustering. The app now enables joint assignment **only when the user selects
2 speakers**, preventing two locally distinct tracks from independently choosing
the same global speaker. It preserves centroid linkage, the trained distance
threshold and minimum cluster size. Auto and all other speaker counts retain the
original policy because the four-speaker check regressed. The two-speaker change
is reapplied after a CUDA-to-CPU recovery.

These short fixtures are diagnostic checks, not a representative accuracy
benchmark. Substantial model errors remain on the harder clips; this correction
does not make speaker identification reliable in every conversation.

## Speaker changes inside whisper.cpp subtitles

The AMD/Metal transcription path previously discarded word timings and assigned
one speaker to each entire subtitle, even when voices changed within it. It now
requests full JSON token timestamps, joins subword tokens into whole words, and
uses the same word-based speaker labeling as Faster-Whisper. The parser handles
UTF-8 characters split between tokens by older whisper.cpp versions. Missing or
invalid word timings retain the original subtitle text and segment-level label.

The public sample can exercise both fixes through the actual AMD inference paths:

```powershell
build/ivrit-speaker-env/Scripts/python.exe scripts/test_speaker_word_labels.py
```

This requires `Models/ggml-large-v3.bin` and uses only the fixed public sample.
It reproduces the old assignment policy, then compares with the revised policy
and word labels. On the 30-second public sample, speaker mismatches fell from
**11 to 2 out of 80 evaluated words**; 13 original subtitles became 17 speaker
spans with the text preserved exactly. This comparison uses ASR word times and
public reference speaker intervals, not manually verified word-level alignment.
It is separate from the audio diarization error rates above.

Regression tests cover the two-local-voices/one-global-label failure, CPU recovery,
backend word-timing requests, subword grouping, split UTF-8, invalid timings, and
text preservation. Native Mac and NVIDIA hardware validation remains outstanding.

## GPU speech filters and measured progress

`engine/speaker_segmentation.py` accelerates SincNet speech filters through
DirectML on Windows or MPS on Apple Silicon. Input normalization and the original
recurrent tracking layers remain on CPU. Centered variance is explicit in the
exported normalization layers; near-silent windows (peak below 1e-4) retain the
CPU path to avoid amplifying numerical differences. Failed GPU batches retry on
CPU, and the status reports the fallback. CUDA still runs the complete model.

An idle-machine, warm comparison used
four CPU threads, mixed windows from the public sample, and the median of three
repetitions per configuration. For 32 ten-second windows: native CPU took
0.278 s, GPU filters plus CPU tracking took 0.180 s, and the full DirectML export
took 1.913 s. At batch 8 these were 0.0686 s, 0.0465 s, and 0.0684 s. These are
model inference timings on this RX 7600M XT, not full-job or cold-start timings.
The implemented split preserved all frame decisions on these inputs. A full
public-sample pipeline comparison produced exactly the same 11 speaker turns
on CPU and the accelerated path.

`scripts/benchmark_speaker_segmentation.py` checks the GPU filters with public
speech, generated noise, silence, and very quiet audio at batch sizes 1/8/32.
All checks passed; the DirectML execution profile confirms GPU work. The full
regression suite passes 136 tests. Native Metal hardware remains unverified.

The progress line displays the current step, percentage, and measured remaining
time. The three steps are speech detection, voice comparison, and transcription.
Estimates reset between steps/retries and exclude pauses; voice-label clustering
is reported as finalization because it exposes no incremental progress. The
transcription estimate starts after speaker processing, so it no longer includes
speaker setup/detection time. Both transcription engines report progress within
chunks. An offscreen UI check verified the line beneath the running status and
the matching percentage bar without reading or changing user settings.

## Next validation

- Use a public, manually labeled Hebrew conversation with at least two speakers.
- Test real conversations through the complete transcription path, including
  word-level speaker changes and TXT/SRT exports.
- Include silences, rapid exchanges, overlapping speech, and multiple chunks.
- Compare Community-1 once its download access is available.
- Run the strict GPU check on native NVIDIA and Apple Silicon hardware.

Sources: [ivrit diarization implementation](https://github.com/ivrit-ai/ivrit-py/blob/master/ivrit/diarization.py),
[official RunPod dependency setup](https://github.com/ivrit-ai/runpod-serverless/blob/main/Dockerfile).
