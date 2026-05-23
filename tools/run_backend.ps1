$ErrorActionPreference = "Stop"
Set-Location "C:\Users\devan\OneDrive\Desktop\deepfake ai"
if (-not (Test-Path ".logs")) {
    New-Item -ItemType Directory -Path ".logs" | Out-Null
}
$command = '"venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 1>".logs\backend.out.log" 2>&1'
cmd.exe /c $command
