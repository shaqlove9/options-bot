@echo off
title Close iobot Dashboard
REM Closes ONLY the local SSH tunnel. The bot keeps running on the server (systemd).
taskkill /FI "WINDOWTITLE eq iobot-tunnel*" /T /F >nul 2>&1
if errorlevel 1 (
    echo No tunnel window found ^(already closed^).
) else (
    echo Tunnel closed.
)
echo The iobot service on the VM is unaffected and keeps running.
timeout /t 2 >nul
