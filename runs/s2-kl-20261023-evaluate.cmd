@echo off
setlocal
cd /d E:\Bomberman
title Bomberman S2-KL - diagnostics then validation
set "S2KL_PYTHON=E:\Bomberman\venv\Scripts\python.exe"
set "S2KL_RUNNER=E:\Bomberman\run_s2_kl_probe.py"
set "S2KL_OUTPUT=E:\Bomberman\runs\s2-kl-20261023"
if not exist "%S2KL_OUTPUT%\arms\control\completed.json" goto missing_training
if not exist "%S2KL_OUTPUT%\arms\anchor\completed.json" goto missing_training
call :stage diagnose
if errorlevel 1 goto failed
call :stage validate
if errorlevel 1 goto failed
call :stage package
if errorlevel 1 goto failed
echo EVALUATION_COMPLETE> "%S2KL_OUTPUT%\logs\launcher-status.txt"
echo Diagnostics, validation and packaging finished. Read report.json.
exit /b 0

:stage
set "S2KL_RESUME="
if "%~1"=="diagnose" if exist "%S2KL_OUTPUT%\diagnostics\" set "S2KL_RESUME=--resume"
if "%~1"=="validate" if exist "%S2KL_OUTPUT%\validation\" set "S2KL_RESUME=--resume"
echo RUNNING_%~1> "%S2KL_OUTPUT%\logs\launcher-status.txt"
echo [%date% %time%] starting %~1 %S2KL_RESUME% >> "%S2KL_OUTPUT%\logs\evaluation-launcher.log"
echo Starting %~1 %S2KL_RESUME%...
"%S2KL_PYTHON%" -u "%S2KL_RUNNER%" %~1 --output "%S2KL_OUTPUT%" %S2KL_RESUME% >> "%S2KL_OUTPUT%\logs\%~1-console.log" 2>&1
set "S2KL_EXIT=%ERRORLEVEL%"
echo [%date% %time%] %~1 exitcode=%S2KL_EXIT% >> "%S2KL_OUTPUT%\logs\evaluation-launcher.log"
exit /b %S2KL_EXIT%

:missing_training
echo Both training arms must be complete before evaluation.
exit /b 1

:failed
echo EVALUATION_STOPPED> "%S2KL_OUTPUT%\logs\launcher-status.txt"
echo Evaluation stopped. Check stage logs; this CMD supports resume.
exit /b 1
