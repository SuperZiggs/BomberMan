@echo off
setlocal
cd /d E:\Bomberman
title Bomberman D-S2 GPU - control then pool
set "DS2_PYTHON=E:\Bomberman\venv\Scripts\python.exe"
set "DS2_RUNNER=E:\Bomberman\run_phase_d_s2.py"
set "DS2_OUTPUT=E:\Bomberman\runs\phaseD-s2-20261022"
echo D-S2: CUDA training, 2000 episodes per arm.
echo Output: %DS2_OUTPUT%
echo Progress stays in this console; details append to control.log and pool.log.
echo.
if not exist "%DS2_OUTPUT%\registration.json" goto missing_registration
call :train_arm control
if errorlevel 1 goto failed
call :train_arm pool
if errorlevel 1 goto failed
echo.
echo Both training arms finished. Validation and test have not been started.
exit /b 0

:train_arm
set "DS2_RESUME="
if exist "%DS2_OUTPUT%\arms\%~1\" set "DS2_RESUME=--resume"
echo Starting %~1 %DS2_RESUME% on registered CUDA device...
echo [%date% %time%] train %~1 %DS2_RESUME%>> "%DS2_OUTPUT%\%~1.log"
"%DS2_PYTHON%" "%DS2_RUNNER%" train --output "%DS2_OUTPUT%" --arm %~1 %DS2_RESUME% >> "%DS2_OUTPUT%\%~1.log" 2>&1
exit /b %errorlevel%

:missing_registration
echo Registration missing. Run the documented register stage first.
exit /b 1

:failed
echo.
echo Training stopped with an error or interruption. Check the arm log.
echo Run this same CMD file again to resume; do not delete registered artifacts.
exit /b 1
