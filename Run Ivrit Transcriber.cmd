@echo off
cd /d "%~dp0"
if not exist "build\ivrit-speaker-env\Scripts\python.exe" (
    echo Setting up the application's Python environment...
    py -3.12 -m venv "build\ivrit-speaker-env"
    if errorlevel 1 goto failed
    goto setup
)
if /i "%~1"=="--setup" goto setup
goto run
:setup
"build\ivrit-speaker-env\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto failed
"build\ivrit-speaker-env\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-speakers.txt
if errorlevel 1 goto failed
"build\ivrit-speaker-env\Scripts\python.exe" scripts\setup_speaker_acceleration.py
if errorlevel 1 goto failed
:run
"build\ivrit-speaker-env\Scripts\python.exe" app.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo The app could not start. Please keep this window open to review the error above.
pause
exit /b 1
