@echo off
setlocal
cd /d "%~dp0"
if not exist "..\venv\Scripts\python.exe" (
  echo Can moi truong Python cua repo tai ..\venv voi Torch va NumPy.
  exit /b 1
)
"..\venv\Scripts\python.exe" -m pip install --upgrade --cache-dir ".pip-cache" --target ".python-deps" -r "requirements.lock.txt"
if errorlevel 1 exit /b 1
call npm.cmd ci --cache ".npm-cache" --no-audit --no-fund
if errorlevel 1 exit /b 1
call npm.cmd run build
if errorlevel 1 exit /b 1
echo Mo http://127.0.0.1:8000 sau khi chay start.cmd
