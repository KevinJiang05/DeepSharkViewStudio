@echo off
setlocal
cd /d "%~dp0" || exit /b 1

set "RUNTIME_CHECK=import numpy, cv2, yaml, PySide6"
for %%I in ("%~dp0..\..\envs\deep-shark-view-studio\Scripts\python.exe") do set "SHARED_PYTHON=%%~fI"

if not defined DEEP_SHARK_PYTHON goto check_venv
if not exist "%DEEP_SHARK_PYTHON%" goto check_venv
"%DEEP_SHARK_PYTHON%" -c "%RUNTIME_CHECK%" >nul 2>&1
if not errorlevel 1 goto run_configured

:check_venv
if not exist ".venv\Scripts\python.exe" goto check_shared
".venv\Scripts\python.exe" -c "%RUNTIME_CHECK%" >nul 2>&1
if not errorlevel 1 goto run_venv

:check_shared
if not exist "%SHARED_PYTHON%" goto check_py
"%SHARED_PYTHON%" -c "%RUNTIME_CHECK%" >nul 2>&1
if not errorlevel 1 goto run_shared

:check_py
py -3 -c "%RUNTIME_CHECK%" >nul 2>&1
if not errorlevel 1 goto run_py
python -c "%RUNTIME_CHECK%" >nul 2>&1
if not errorlevel 1 goto run_python

echo DeepShark View Studio could not find a Python environment with all runtime dependencies. 1>&2
echo Set DEEP_SHARK_PYTHON, create .venv, or install requirements.txt, then retry. 1>&2
if "%~1"=="" pause
exit /b 2

:run_configured
"%DEEP_SHARK_PYTHON%" app.py %*
goto finish

:run_venv
".venv\Scripts\python.exe" app.py %*
goto finish

:run_shared
"%SHARED_PYTHON%" app.py %*
goto finish

:run_py
py -3 app.py %*
goto finish

:run_python
python app.py %*
goto finish

:finish
set "APP_EXIT_CODE=%errorlevel%"
if "%APP_EXIT_CODE%"=="0" exit /b 0
echo DeepShark View Studio exited with code %APP_EXIT_CODE%. 1>&2
if "%~1"=="" pause
exit /b %APP_EXIT_CODE%
