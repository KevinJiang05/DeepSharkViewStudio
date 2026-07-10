@echo off
setlocal
cd /d "%~dp0" || exit /b 1

if not exist ".venv\Scripts\python.exe" goto check_py
".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if not errorlevel 1 goto run_venv

:check_py
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 goto run_py
python -c "import sys" >nul 2>&1
if not errorlevel 1 goto run_python

echo DeepShark View Studio could not find Python. 1>&2
echo Create .venv or install Python 3, then retry. 1>&2
exit /b 9009

:run_venv
".venv\Scripts\python.exe" app.py %*
exit /b %errorlevel%

:run_py
py -3 app.py %*
exit /b %errorlevel%

:run_python
python app.py %*
exit /b %errorlevel%
