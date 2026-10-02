"""Verify a built executable using its own runtime, GUI, and media loader."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--report', type=Path, default=Path('logs/exe-smoke-report.json'))
    args = parser.parse_args()
    executable = args.executable.resolve()
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    try:
        result = subprocess.run(
            [str(executable), '--smoke-test-report', str(report)],
            cwd=executable.parent,
            env=dict(os.environ, QT_QPA_PLATFORM='windows' if os.name == 'nt' else 'offscreen'),
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f'Executable smoke check failed: {exc}')
    if not report.is_file():
        raise SystemExit(f'Executable produced no smoke report (exit {result.returncode}).')
    data = json.loads(report.read_text(encoding='utf-8'))
    if result.returncode != 0 or data.get('ok') is not True:
        raise SystemExit(f'Executable smoke check failed: {data}')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
