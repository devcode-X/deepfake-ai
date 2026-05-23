@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem One-click launcher for the DeepFake Detector app on Windows.
rem It creates/reuses the local venv, installs requirements into that venv,
rem starts FastAPI on :8000 and Streamlit on :8501, and prints the URLs.

cd /d "%~dp0"

set "BACKEND_PORT=8000"
set "FRONTEND_PORT=8501"
set "VENV_DIR=%CD%\venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo.
echo DeepFake Detector Launcher
echo Workspace: %CD%
echo.

if not exist "backend\.env" (
    echo [WARN] backend\.env was not found.
    echo [WARN] The backend needs a Hugging Face token in backend\.env to run correctly.
    echo.
)

echo [1/5] Checking virtual environment...
if not exist "%VENV_PY%" (
    echo Creating venv in "%VENV_DIR%"...
    where py >nul 2>&1
    if !errorlevel! equ 0 (
        py -3 -m venv "%VENV_DIR%"
    ) else (
        python -m venv "%VENV_DIR%"
    )
    if errorlevel 1 goto :fail
) else (
    echo Reusing existing venv.
)

echo.
echo [2/5] Ensuring pip is available in venv...
"%VENV_PY%" -m ensurepip --upgrade >nul 2>&1
if errorlevel 1 (
    echo [WARN] ensurepip reported an issue. Continuing with the current venv pip state.
)
"%VENV_PY%" -m pip --version >nul 2>&1
if errorlevel 1 goto :fail

echo.
echo [3/5] Installing or updating project requirements in venv...
"%VENV_PY%" -m pip --disable-pip-version-check install -r backend\requirements.txt -r frontend\requirements.txt
if errorlevel 1 goto :fail

rem --- Detect LAN IP for display ---
set "LAN_IP="
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /i "IPv4"') do (
    for /f "tokens=*" %%B in ("%%A") do (
        if not defined LAN_IP set "LAN_IP=%%B"
    )
)
if not defined LAN_IP set "LAN_IP=localhost"

echo.
echo [4/5] Starting backend and frontend if needed...

call :check_port %BACKEND_PORT% BACKEND_PID
if defined BACKEND_PID (
    echo Backend already listening on port %BACKEND_PORT% ^(PID !BACKEND_PID!^).
) else (
    echo Starting backend on http://0.0.0.0:%BACKEND_PORT% ...
    start "Deepfake Backend" cmd /k "cd /d ""%CD%"" && title Deepfake Backend && ""%VENV_PY%"" -m uvicorn backend.main:app --host 0.0.0.0 --port %BACKEND_PORT%"
)

call :check_port %FRONTEND_PORT% FRONTEND_PID
if defined FRONTEND_PID (
    echo Frontend already listening on port %FRONTEND_PORT% ^(PID !FRONTEND_PID!^).
) else (
    echo Starting Streamlit on http://0.0.0.0:%FRONTEND_PORT% ...
    start "Deepfake Frontend" cmd /k "cd /d ""%CD%"" && title Deepfake Frontend && ""%VENV_PY%"" -m streamlit run frontend/app.py --server.address 0.0.0.0 --server.port %FRONTEND_PORT%"
)

echo.
echo [5/5] App URLs
echo.
echo   Local URLs:
echo     Streamlit UI : http://localhost:%FRONTEND_PORT%
echo     Backend API  : http://localhost:%BACKEND_PORT%
echo     API Docs     : http://localhost:%BACKEND_PORT%/docs
echo.
echo   LAN / External URLs (use from other devices on same network):
echo     Streamlit UI : http://!LAN_IP!:%FRONTEND_PORT%
echo     Backend API  : http://!LAN_IP!:%BACKEND_PORT%
echo     API Docs     : http://!LAN_IP!:%BACKEND_PORT%/docs
echo.
echo If one of the app windows shows an error, fix that window and re-run this launcher.
goto :eof

:check_port
set "%~2="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%~1" ^| findstr "LISTENING"') do (
    set "%~2=%%P"
    goto :eof
)
goto :eof

:fail
echo.
echo Launcher failed. Review the output above and fix the reported issue.
exit /b 1
