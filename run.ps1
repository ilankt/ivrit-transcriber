# Get the directory where this script is located
$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
cd $PSScriptRoot

# Activate and Run
& ".\.venv\Scripts\Activate.ps1"
python app.py