@echo off
setlocal
cd /d "%~dp0"
if not exist "..\venv\Scripts\python.exe" (
  echo Khong tim thay Python co Torch tai ..\venv\Scripts\python.exe
  exit /b 1
)
if not exist "dist\index.html" (
  echo Chay setup.cmd truoc de cai dependency va build giao dien.
  exit /b 1
)
"..\venv\Scripts\python.exe" run.py %*
