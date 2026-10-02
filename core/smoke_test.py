"""Opt-in packaged-app checks; never run during normal application startup."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import traceback
import wave

from PySide6.QtCore import QTimer


def schedule_smoke_test(app, window, file_load_worker, report_path):
    """Exercise the actual GUI, bundled VAD, and media loader, then exit."""
    def check():
        report = {"ok": False, "checks": []}
        try:
            report_file = Path(report_path).resolve()
            report_file.parent.mkdir(parents=True, exist_ok=True)
            if not window.isVisible():
                raise RuntimeError("Main window was not created and shown.")
            report["window_title"] = window.windowTitle()
            if not window.grab().save(str(report_file.with_suffix('.png'))):
                raise RuntimeError("Could not render the main window.")
            report["checks"].append("Main window created and rendered")

            import numpy as np
            from faster_whisper.vad import get_speech_timestamps
            get_speech_timestamps(np.zeros(16000, dtype=np.float32))
            report["checks"].append("Bundled voice detection ran successfully")

            from engine.ffmpeg_helper import _find_executable, _POPEN_EXTRA_KWARGS
            with tempfile.TemporaryDirectory(prefix='ivrit_smoke_') as directory:
                for extension in ('m4a', 'mp4'):
                    source = os.path.join(directory, f'sample.{extension}')
                    args = [
                        _find_executable('ffmpeg'), '-v', 'error', '-y',
                        '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
                    ]
                    if extension == 'mp4':
                        args += ['-f', 'lavfi', '-i', 'color=size=32x32:duration=1', '-c:v', 'mpeg4']
                    args += ['-c:a', 'aac', source]
                    subprocess.run(args, check=True, capture_output=True, timeout=15, **_POPEN_EXTRA_KWARGS)
                    jobs, errors = [], []
                    worker = file_load_worker(source)
                    worker.loaded.connect(jobs.append)
                    worker.error.connect(errors.append)
                    worker.run()
                    try:
                        if errors or len(jobs) != 1 or not jobs[0].tasks:
                            raise RuntimeError(f'{extension} load failed: {errors}')
                        with wave.open(jobs[0].tasks[0].chunk_path, 'rb') as audio:
                            if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) != (1, 16000, 2):
                                raise RuntimeError(f'{extension} did not decode to 16 kHz mono PCM')
                            if audio.getnframes() < 15000:
                                raise RuntimeError(f'{extension} decoded audio is too short')
                        report["checks"].append(f'{extension.upper()} decoded through the packaged file loader')
                    finally:
                        for job in jobs:
                            if job.temp_dir:
                                shutil.rmtree(job.temp_dir)
            report["ok"] = True
        except Exception:
            report["error"] = traceback.format_exc()
        finally:
            # A failed import before this callback cannot produce a success report.
            try:
                Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
            finally:
                # Avoid closeEvent: smoke checks must not save user preferences.
                app.exit(0 if report["ok"] else 1)

    QTimer.singleShot(0, check)
