@echo off
REM run_all.bat - Start all three services simultaneously (Windows)
REM Usage: run_all.bat

echo.
echo ========================================
echo MOBY Edge System - Starting All Services
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: python not found. Please install Python 3.
    pause
    exit /b 1
)

echo [1/3] Starting motor_PdM.py (Motor + IR Sensor)...
start "Motor PdM" cmd /k python motor_PdM.py
timeout /t 2 /nobreak

echo [2/3] Starting sensor_final.py (Multi-Sensor Publisher)...
start "Sensor Final" cmd /k python src\sensor_final.py
timeout /t 2 /nobreak

echo [3/3] Starting inference_worker.py (Anomaly Detection)...
start "Inference Worker" cmd /k python src\inference_worker.py
timeout /t 2 /nobreak

echo.
echo ========================================
echo OK All services started!
echo ========================================
echo.
echo Three windows should appear with each service running.
echo Close each window individually to stop that service.
echo.
pause
