#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py - Unified launcher for MOBY Edge System
Starts all three services simultaneously with logging and monitoring.

Usage:
    python run_all.py
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

# Process tracking
processes = {}
script_dir = Path(__file__).parent.absolute()

def log(msg, level="INFO"):
    """Print timestamped log message."""
    timestamp = time.strftime("%H:%M:%S")
    icon = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARN": "⚠️ ",
    }.get(level, "")
    print(f"[{timestamp}] {icon} {msg}")

def start_service(name, script_path, description):
    """Start a Python service in a separate process."""
    full_path = script_dir / script_path
    
    if not full_path.exists():
        log(f"{description}: Script not found at {full_path}", "ERROR")
        return None
    
    try:
        log(f"Starting {description}...", "INFO")
        proc = subprocess.Popen(
            [sys.executable, str(full_path)],
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        processes[name] = proc
        log(f"{description} started (PID: {proc.pid})", "SUCCESS")
        return proc
    except Exception as e:
        log(f"{description} failed to start: {e}", "ERROR")
        return None

def monitor_processes():
    """Monitor running processes and report if any exit."""
    while True:
        time.sleep(1)
        for name, proc in list(processes.items()):
            if proc.poll() is not None:
                # Process exited
                retcode = proc.returncode
                if retcode == 0:
                    log(f"{name} exited normally", "INFO")
                else:
                    log(f"{name} exited with code {retcode}", "WARN")

def stop_all_services():
    """Gracefully stop all running services."""
    log("Stopping all services...", "INFO")
    
    for name, proc in processes.items():
        try:
            log(f"Terminating {name} (PID: {proc.pid})...", "INFO")
            proc.terminate()
        except Exception as e:
            log(f"Failed to terminate {name}: {e}", "WARN")
    
    # Wait for graceful shutdown
    time.sleep(2)
    
    # Force kill if still running
    for name, proc in processes.items():
        if proc.poll() is None:
            try:
                log(f"Force killing {name}...", "WARN")
                proc.kill()
            except Exception as e:
                log(f"Failed to kill {name}: {e}", "ERROR")
    
    log("All services stopped", "SUCCESS")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    log("Interrupt received", "WARN")
    stop_all_services()
    sys.exit(0)

def main():
    """Main launcher function."""
    log("=" * 50, "INFO")
    log("MOBY Edge System - Unified Launcher", "SUCCESS")
    log("=" * 50, "INFO")
    log("")
    
    # Check Python version
    if sys.version_info < (3, 7):
        log(f"Python 3.7+ required (current: {sys.version})", "ERROR")
        sys.exit(1)
    
    # Check MQTT broker connectivity (optional but recommended)
    log("Checking MQTT broker...", "INFO")
    try:
        import paho.mqtt.client as mqtt
        client = mqtt.Client(client_id="launcher_test")
        result = client.connect("192.168.80.192", 1883, 5)
        if result == 0:
            log("MQTT broker reachable at 192.168.80.192:1883", "SUCCESS")
        else:
            log(f"MQTT broker returned code {result} (may still work)", "WARN")
        client.disconnect()
    except Exception as e:
        log(f"Cannot reach MQTT broker: {e}", "WARN")
        log("Services will still try to reconnect automatically", "INFO")
    
    log("")
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Start all three services
    services = [
        ("motor_pdm", "motor_PdM.py", "Motor PdM (Motor + IR Sensor)"),
        ("sensor_final", "src/sensor_final.py", "Sensor Final (Multi-Sensor Publisher)"),
        ("inference_worker", "src/inference_worker.py", "Inference Worker (Anomaly Detection)"),
    ]
    
    started_count = 0
    for name, script, description in services:
        start_service(name, script, description)
        started_count += len(processes)
        time.sleep(1.5)  # Stagger startup
    
    log("")
    if len(processes) == len(services):
        log(f"✅ All {len(services)} services started successfully!", "SUCCESS")
    else:
        log(f"⚠️  Only {len(processes)}/{len(services)} services started", "WARN")
    
    log("")
    log("Running processes:", "INFO")
    for name, proc in processes.items():
        log(f"  {name:20} (PID: {proc.pid})", "INFO")
    
    log("")
    log("Press Ctrl+C to stop all services", "INFO")
    log("")
    
    # Monitor processes
    try:
        while True:
            time.sleep(1)
            # Check if any process has exited
            for name, proc in list(processes.items()):
                if proc.poll() is not None:
                    log(f"{name} has exited (code: {proc.returncode})", "WARN")
                    # Could restart here if desired
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
