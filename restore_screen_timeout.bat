@echo off
title FreeSight-OS - Restore Display Timeout Defaults
cd /d "%~dp0"
echo ================================================================
echo  FreeSight-OS: Restore Display Defaults
echo ================================================================
echo.
echo [*] Restoring default monitor timeout (15 mins AC, 10 mins DC)...
powercfg /change monitor-timeout-ac 15
powercfg /change monitor-timeout-dc 10
echo [*] Restoring default system standby timeout (30 mins AC, 15 mins DC)...
powercfg /change standby-timeout-ac 30
powercfg /change standby-timeout-dc 15
echo.
echo [SUCCESS] Windows default screen and sleep timeouts have been restored.
echo.
pause
