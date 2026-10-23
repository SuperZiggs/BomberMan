@echo off
setlocal
cd /d E:\Bomberman
title Bomberman S2-KL GPU - control then anchor
set "S2KL_PYTHON=E:\Bomberman\venv\Scripts\python.exe"
set "S2KL_RUNNER=E:\Bomberman\run_s2_kl_probe.py"
set "S2KL_OUTPUT=E:\Bomberman\runs\s2-kl-20261023"
echo S2-KL: CUDA training, 500 episodes per arm.
echo Output: %S2KL_OUTPUT%
echo Progress stays in this console; details append to logs.
if not exist "%S2KL_OUTPUT%\registration.json" goto missing_registration
call :train_arm control
if errorlevel 1 goto failed
call :train_arm anchor
if errorlevel 1 goto failed
echo TRAIN_COMPLETE> "%S2KL_OUTPUT%\logs\launcher-status.txt"
echo Both training arms finished. Diagnostics and validation are next.
exit /b 0

:train_arm
set "S2KL_RESUME="
if exist "%S2KL_OUTPUT%\arms\%~1\" set "S2KL_RESUME=--resume"
echo TRAINING_%~1> "%S2KL_OUTPUT%\logs\launcher-status.txt"
echo [%date% %time%] train %~1 %S2KL_RESUME%>> "%S2KL_OUTPUT%\logs\launcher.log"
echo Starting %~1 %S2KL_RESUME% on registered CUDA device...
"%S2KL_PYTHON%" -u "%S2KL_RUNNER%" train --output "%S2KL_OUTPUT%" --arm %~1 %S2KL_RESUME% >> "%S2KL_OUTPUT%\logs\%~1-console.log" 2>&1
set "S2KL_EXIT=%ERRORLEVEL%"
echo [%date% %time%] train %~1 exit %S2KL_EXIT%>> "%S2KL_OUTPUT%\logs\launcher.log"
exit /b %S2KL_EXIT%

:missing_registration
echo Registration missing. Run the register stage first.
exit /b 1

:failed
echo TRAINING_STOPPED> "%S2KL_OUTPUT%\logs\launcher-status.txt"
echo Training stopped. Check logs; run this same CMD file to resume.
exit /b 1
