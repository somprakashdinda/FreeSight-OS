@echo off
title FreeSight-OS Real-Time Browser Studio
cd /d "%~dp0"
echo ================================================================
echo  Starting FreeSight-OS Real-Time Browser Studio...
echo  Opening http://localhost:8080 in your browser...
echo ================================================================
.\.venv\Scripts\python.exe web_server.py
pause
