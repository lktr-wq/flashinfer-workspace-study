@echo off
chcp 65001 >nul
title FlashInfer S03-M06 correctness
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1"
echo Exit code: %errorlevel%
pause
