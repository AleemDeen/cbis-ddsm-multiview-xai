@echo off
setlocal enabledelayedexpansion
title Mammogram XAI - Launcher

echo.
echo ================================================
echo   Mammogram XAI - Starting up...
echo ================================================
echo.

:: ------------------------------------------------
:: 1. Check Python is installed
:: ------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on your system.
    echo.
    echo Please install Python 3.10 or later from https://www.python.org/downloads/
    echo Make sure to tick "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo [OK] Found %PYTHON_VERSION%

:: ------------------------------------------------
:: 2. Check Node.js / npm is installed
:: ------------------------------------------------
npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js was not found on your system.
    echo.
    echo Please install Node.js from https://nodejs.org/
    echo The LTS version is recommended.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('npm --version 2^>^&1') do set NPM_VERSION=%%v
echo [OK] Found npm v%NPM_VERSION%

:: ------------------------------------------------
:: 3. Create virtual environment if it doesn't exist
:: ------------------------------------------------
if not exist ".venv" (
    echo.
    echo [SETUP] Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

:: ------------------------------------------------
:: 4. Activate virtual environment
:: ------------------------------------------------
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.

:: ------------------------------------------------
:: 5. Install / update Python dependencies
:: ------------------------------------------------
echo.
echo [SETUP] Installing Python dependencies (this may take a moment on first run)...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)
echo [OK] Python dependencies ready.

:: ------------------------------------------------
:: 6. Install frontend dependencies
:: ------------------------------------------------
echo.
echo [SETUP] Installing frontend dependencies...
cd src\frontend
npm install --silent
if errorlevel 1 (
    echo [ERROR] Failed to install frontend dependencies.
    cd ..\..
    pause
    exit /b 1
)
cd ..\..
echo [OK] Frontend dependencies ready.

:: ------------------------------------------------
:: 7. Start the FastAPI backend in a new window
:: ------------------------------------------------
echo.
echo [START] Starting FastAPI backend on http://localhost:8000 ...
start "Mammogram XAI - Backend" cmd /k "call .venv\Scripts\activate.bat && uvicorn src.api.server:app --port 8000"

:: ------------------------------------------------
:: 8. Give the backend a moment to bind its port
:: ------------------------------------------------
timeout /t 3 /nobreak >nul

:: ------------------------------------------------
:: 9. Start the Vite frontend in a new window
:: ------------------------------------------------
echo [START] Starting frontend on http://localhost:5173 ...
start "Mammogram XAI - Frontend" cmd /k "cd src\frontend && npm run dev"

:: ------------------------------------------------
:: 10. Wait for frontend to be ready then open browser
:: ------------------------------------------------
echo.
echo [WAIT] Waiting for servers to be ready...
timeout /t 4 /nobreak >nul

echo [OK] Opening browser...
start http://localhost:5173

echo.
echo ================================================
echo   Both servers are running.
echo.
echo   Backend:   http://localhost:8000
echo   Frontend:  http://localhost:5173
echo   API docs:  http://localhost:8000/docs
echo.
echo   Close the two server windows to shut down.
echo ================================================
echo.
pause
endlocal
