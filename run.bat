@echo off
title Direct Ocular Precision Controller (DOPC v7.0) - Live Dashboard
cd /d "%~dp0"
echo ================================================================
echo  Starting Direct Ocular Precision Controller (DOPC v7.0)...
echo  Configuring continuous display keep-awake...
echo ================================================================
powercfg /change monitor-timeout-ac 0 >nul 2>&1
powercfg /change monitor-timeout-dc 0 >nul 2>&1
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change standby-timeout-dc 0 >nul 2>&1
if "%~1"=="" (
    .\.venv\Scripts\python.exe main.py --mode 1
) else (
    .\.venv\Scripts\python.exe main.py %*
)
pause
