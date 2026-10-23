@echo off
chcp 65001 >nul
title FlashInfer S03-M06 stop and save
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
echo Exit code: %errorlevel%
pause
