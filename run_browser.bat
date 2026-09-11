@echo off
title FreeSight-OS Real-Time Browser Studio
cd /d "%~dp0"
echo ================================================================
echo  Starting FreeSight-OS Real-Time Browser Studio...
echo  Configuring continuous display keep-awake...
echo  Opening http://localhost:8080 in your browser...
echo ================================================================
powercfg /change monitor-timeout-ac 0 >nul 2>&1
powercfg /change monitor-timeout-dc 0 >nul 2>&1
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change standby-timeout-dc 0 >nul 2>&1
start http://localhost:8080
.\.venv\Scripts\python.exe web_server.py
pause
