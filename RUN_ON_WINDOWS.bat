@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo CSE570 Bank Marketing ML Project - Windows
echo ==============================================

where py >nul 2>nul
if %errorlevel%==0 (
    set PY=py -3
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python was not found. Install Python 3.11+ and enable PATH.
        pause
        exit /b 1
    )
    set PY=python
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 goto failed
)

echo [2/3] Installing required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo [3/3] Running full analysis and model training...
".venv\Scripts\python.exe" src\train.py
if errorlevel 1 goto failed

echo.
echo SUCCESS: Project execution finished.
echo Open the outputs folder to inspect metrics, tables and charts.
echo Then open analysis_walkthrough.ipynb for the guided analysis.
pause
exit /b 0

:failed
echo.
echo ERROR: A step failed. Read the message above, check your internet connection,
echo and verify Python is installed. Do not close this window before noting the error.
pause
exit /b 1
