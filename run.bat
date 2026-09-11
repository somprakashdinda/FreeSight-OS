@echo off
title Direct Ocular Precision Controller (DOPC v7.0) - Live Dashboard
cd /d "%~dp0"
echo ================================================================
echo  Starting Direct Ocular Precision Controller (DOPC v7.0)...
echo ================================================================
if "%~1"=="" (
    .\.venv\Scripts\python.exe main.py --mode 1
) else (
    .\.venv\Scripts\python.exe main.py %*
)
pause
