@echo off
title Eye-Tracking Host OS Control - Live Dashboard
cd /d "%~dp0"
echo ===================================================
echo Starting Eye-Tracking Host OS Control (Mode 1)...
echo ===================================================
.\.venv\Scripts\python.exe main.py --mode 1
pause
