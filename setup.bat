@echo off
echo Soccer Tracker - UV Setup Script
echo ====================================

echo.
echo 1. Checking if UV is installed...
uv --version >nul 2>&1
if errorlevel 1 (
    echo UV is not installed. Installing UV...
    pip install uv
    if errorlevel 1 (
        echo Error: Failed to install UV
        exit /b 1
    )
)

echo.
echo 2. Creating virtual environment...
if exist .venv (
    echo .venv directory already exists. Skipping creation.
) else (
    uv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        exit /b 1
    )
)

echo.
echo 3. Installing project dependencies...
.venv\Scripts\uv.exe pip install -e .
if errorlevel 1 (
    echo Error: Failed to install dependencies
    exit /b 1
)

echo.
echo ====================================
echo Setup completed successfully!
echo ====================================
echo.
echo To activate the virtual environment, run:
echo   .venv\Scripts\activate
echo.
echo Then you can run the tracker with:
echo   python main.py
echo ====================================
pause
