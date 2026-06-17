@echo off
title iobot-tunnel
REM Opens an SSH tunnel to the iobot read-only dashboard on the VM, then the browser.
REM Key expected at %USERPROFILE%\Desktop\Options-bot.pem  (same key as before).
set KEY=%USERPROFILE%\Desktop\Options-bot.pem
set HOSTV=ubuntu@3.87.108.0

echo Opening SSH tunnel to iobot dashboard (localhost:8501)...
start "iobot-tunnel" ssh -i "%KEY%" -N -L 8501:localhost:8501 %HOSTV%

echo Waiting for the tunnel to come up...
timeout /t 4 /nobreak >nul

start "" http://localhost:8501
echo.
echo Dashboard: http://localhost:8501
echo Keep the "iobot-tunnel" window open while viewing. Run "Close iobot Dashboard.bat" when done.
timeout /t 3 >nul
