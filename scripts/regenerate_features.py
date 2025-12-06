import os
import sys
sys.path.append(os.getcwd())
import glob
import pandas as pd
from src.feature_extractor import process_multi_sensor_files, OUTPUT_DIR

def regenerate_all():
    data_dir = "data"
    
    # Define dataset pairings (Accel/Gyro file -> IR Counter file)
    # logic: find accel files, look for matching IR file with same prefix/suffix
    
    accel_files = glob.glob(os.path.join(data_dir, "*_accel_gyro_*.csv"))
    
    for accel_path in accel_files:
        basename = os.path.basename(accel_path)
        # construct expected ir filename
        # e.g. 1124_accel_gyro_normal.csv -> 1124_IRcounter_normal.csv
        ir_name = basename.replace("accel_gyro", "IRcounter")
        ir_path = os.path.join(data_dir, ir_name)
        
        if not os.path.exists(ir_path):
            print(f"Skipping {basename}: Matching IR file {ir_name} not found.")
            continue
            
        print(f"\nProcessing Group: {basename}")
        
        input_files = {
            'accel_gyro': accel_path,
            'ir_counter': ir_path
        }
        
        # Output filename
        out_name = basename.replace("accel_gyro_", "").replace(".csv", "_features.csv")
        # e.g. 1124_normal_features.csv or just prefix based
        # Let's try to keep it identifiable. 
        # 1120_accel_gyro_s1_red.csv -> 1120_s1_red_features.csv
        
        try:
            df = process_multi_sensor_files(
                input_files,
                resample_rate='78.125ms', # 12.8Hz
                window_size=10.0,
                window_overlap=5.0
            )
            
            if not df.empty:
                out_path = os.path.join(OUTPUT_DIR, out_name)
                df.to_csv(out_path, index=False)
                print(f"Saved: {out_path} ({len(df)} rows)")
        except Exception as e:
            print(f"Failed to process {basename}: {e}")

if __name__ == "__main__":
    regenerate_all()
