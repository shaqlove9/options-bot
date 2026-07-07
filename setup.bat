@echo off
title Options Bot Setup
cd /d "%~dp0"

echo.
echo ========================================
echo   Options Bot — First-Time Setup
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo Download it from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

:: Create venv if missing
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment already exists — skipping.
)

:: Install dependencies
echo [2/4] Installing dependencies...
".venv\Scripts\pip.exe" install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: pip install failed. Check your internet connection.
    pause
    exit /b 1
)

:: Copy .env if missing
if not exist ".env" (
    echo [3/4] Creating .env from template...
    copy .env.example .env >nul
    echo.
    echo  *** IMPORTANT: Open .env in a text editor and fill in your
    echo      ALPACA_API_KEY and ALPACA_SECRET_KEY before running the bot.
    echo      Get paper keys at: https://app.alpaca.markets
    echo.
) else (
    echo [3/4] .env already exists — skipping.
)

:: Initialize trade database
echo [4/4] Initializing trade database...
".venv\Scripts\python.exe" -c "import data.trade_store as ts; ts.init(); print('trades.db ready')"

echo.
echo ========================================
echo   Setup complete!
echo ========================================
echo.
echo   Next steps:
echo   1. Edit .env with your Alpaca API keys
echo   2. Double-click "Launch Options Bot.bat" to start the dashboard
echo   3. Press Start in the dashboard to begin paper trading
echo.

pause
