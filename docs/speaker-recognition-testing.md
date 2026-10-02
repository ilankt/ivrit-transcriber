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
The full automated suite passes 66 tests. This still does not validate accuracy
on real Hebrew conversations.

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

## Next validation

- Use a short, manually labeled Hebrew conversation with at least two speakers.
- Test the complete transcription path, including word-level speaker changes
  and TXT/SRT exports, with ivrit's wrapper.
- Include silences, rapid exchanges, overlapping speech, and multiple chunks.
- Compare Community-1 once its download access is available.
- Validate on a native Mac before claiming Mac speaker-inference compatibility.

Sources: [ivrit diarization implementation](https://github.com/ivrit-ai/ivrit-py/blob/master/ivrit/diarization.py),
[official RunPod dependency setup](https://github.com/ivrit-ai/runpod-serverless/blob/main/Dockerfile).
