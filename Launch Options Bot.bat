@echo off
title Options Bot Dashboard
cd /d "%~dp0"

:: Create venv if missing
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is not installed or not on PATH.
        echo Download it from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Install dependencies
echo Installing dependencies...
".venv\Scripts\pip.exe" install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: pip install failed. Check your internet connection.
    pause
    exit /b 1
)

:: Create .env from template if missing
if not exist ".env" (
    copy .env.example .env >nul
    echo.
    echo ========================================
    echo   .env file created from template.
    echo   Open .env and fill in your API keys
    echo   before pressing Start in the dashboard.
    echo   Get paper keys at: https://app.alpaca.markets
    echo ========================================
    echo.
)

:: Initialize trade database
".venv\Scripts\python.exe" -c "import data.trade_store as ts; ts.init()" 2>nul

:: Launch dashboard
".venv\Scripts\python.exe" -m streamlit run app.py
pause
