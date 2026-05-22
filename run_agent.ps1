$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m pip install -r agent/requirements.txt
python agent/server.py
