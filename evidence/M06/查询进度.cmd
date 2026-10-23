@echo off
chcp 65001 >nul
title FlashInfer S03-M06 progress
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0status.ps1"
echo Exit code: %errorlevel%
pause
