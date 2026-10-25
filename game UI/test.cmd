@echo off
setlocal
cd /d "%~dp0"
"..\venv\Scripts\python.exe" tools\verify.py
exit /b %errorlevel%
