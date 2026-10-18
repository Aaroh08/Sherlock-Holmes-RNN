@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please follow the installation steps in README.md first.
    pause
    exit /b 1
)
echo Baker Street Text Lab: http://127.0.0.1:5001
".venv\Scripts\python.exe" app.py
pause
