$ErrorActionPreference = "Stop"
Set-Location "C:\Users\devan\OneDrive\Desktop\deepfake ai"
if (-not (Test-Path ".logs")) {
    New-Item -ItemType Directory -Path ".logs" | Out-Null
}
$command = '"venv\Scripts\python.exe" -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true 1>".logs\frontend.out.log" 2>&1'
cmd.exe /c $command
