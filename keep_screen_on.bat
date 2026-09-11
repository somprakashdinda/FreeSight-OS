@echo off
title FreeSight-OS - Continuous Display Keep-Awake
cd /d "%~dp0"
echo ================================================================
echo  FreeSight-OS: Display Sleep Prevention Utility
echo ================================================================
echo.
echo [*] Disabling Windows monitor timeout (AC ^& Battery)...
powercfg /change monitor-timeout-ac 0
powercfg /change monitor-timeout-dc 0
echo [*] Disabling Windows system standby sleep (AC ^& Battery)...
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
echo.
echo [SUCCESS] Screen timeout is now set to NEVER.
echo Your display will remain active continuously.
echo (To restore defaults later, run restore_screen_timeout.bat)
echo.
pause
