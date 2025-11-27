#!/bin/bash
# run_all.sh - Start all three services simultaneously
# Usage: bash run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "MOBY Edge System - Starting All Services"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3."
    exit 1
fi

echo "[1/3] Starting motor_PdM.py (Motor + IR Sensor)..."
python3 motor_PdM.py &
PID_MOTOR=$!
echo "      Motor process PID: $PID_MOTOR"
sleep 2

echo "[2/3] Starting sensor_final.py (Multi-Sensor Publisher)..."
python3 src/sensor_final.py &
PID_SENSOR=$!
echo "      Sensor process PID: $PID_SENSOR"
sleep 2

echo "[3/3] Starting inference_worker.py (Anomaly Detection)..."
python3 src/inference_worker.py &
PID_INFERENCE=$!
echo "      Inference process PID: $PID_INFERENCE"
sleep 2

echo ""
echo "========================================"
echo "✅ All services started successfully!"
echo "========================================"
echo ""
echo "Process IDs:"
echo "  Motor PdM:        $PID_MOTOR"
echo "  Sensor Final:     $PID_SENSOR"
echo "  Inference Worker: $PID_INFERENCE"
echo ""
echo "To stop all services, press Ctrl+C"
echo "(or run: kill $PID_MOTOR $PID_SENSOR $PID_INFERENCE)"
echo ""

# Wait for all processes
wait_and_cleanup() {
    echo ""
    echo "Stopping all services..."
    kill $PID_MOTOR 2>/dev/null || true
    kill $PID_SENSOR 2>/dev/null || true
    kill $PID_INFERENCE 2>/dev/null || true
    sleep 1
    echo "✅ All services stopped."
    exit 0
}

# Trap Ctrl+C
trap wait_and_cleanup SIGINT SIGTERM

# Wait for any process to exit
wait -n
wait_and_cleanup
