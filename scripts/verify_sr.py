
import os
import pandas as pd
import numpy as np

def check_sr(filepath):
    print(f"Checking {os.path.basename(filepath)}...")
    try:
        # Read first 3000 lines to estimate SR (InfluxDB format is verbose)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [next(f) for _ in range(3000)]
            
        # Minimal parsing for InfluxDB export CSV
        # Look for header starting with #group or just _time
        data_lines = []
        header = None
        
        for line in lines:
            if line.startswith("#"): continue
            if "_time" in line and header is None:
                header = line.strip().split(',')
                continue
            if header and line.strip():
                data_lines.append(line.strip().split(','))
                
        if not header or not data_lines:
            print("  Could not parse CSV structure")
            return

        df = pd.DataFrame(data_lines, columns=header)
        
        # InfluxDB often has multiple rows per timestamp (one per field), need to pivot or count unique times
        df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
        df = df.dropna(subset=['_time'])
        
        # Count unique timestamps
        unique_times = df['_time'].unique()
        unique_times = np.sort(unique_times)
        
        if len(unique_times) < 2:
            print("  Not enough time data")
            return
            
        # Calculate diffs
        series = pd.Series(unique_times)
        diffs = series.diff().dt.total_seconds().dropna()
        
        # Filter large gaps (packet loss)
        median_dt = diffs.median()
        
        if median_dt > 0:
            sr = 1.0 / median_dt
            print(f"  Estimated SR (Median): {sr:.2f} Hz (dt: {median_dt*1000:.2f} ms)")
        else:
            print(f"  Invalid median dt: {median_dt}")

    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    files = [
        "data/1205_accel_gyro_normal.csv",
        "data/1120_accel_gyro_s1_red.csv" 
    ]
    for f in files:
        if os.path.exists(f):
            check_sr(f)
        else:
            print(f"  File not found: {f}")
