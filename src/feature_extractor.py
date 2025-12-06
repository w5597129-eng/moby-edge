
"""
Feature Extraction from Sensor Data (InfluxDB Format)
Version 19 (FFT High-Frequency):
- Optimized for 100Hz sampling
- Window: 1.0s (100 samples)
- Overlap: 0.5s (50 samples)
- Target Features: 12 Features
- Accel (8): VectorRMS, PC1_PeakToPeak, PC1_DominantFreq, PC1_BandEnergy_Low, PC1_BandEnergy_High, PC1_SpectralEntropy, PC1_RMSF, PC1_VarianceRatio
- Gyro (3): STD_X, STD_Y, STD_Z
- IR (1): AvgCycleTime

Author: WISE Team, Project MOBY
Date: 2025-12-06
"""

import os
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =====================================
# 설정
# =====================================

FEATURE_CONFIG_V19 = {
    'accel': [
        'VectorRMS',
        'PC1_PeakToPeak',
        'PC1_DominantFreq',   # FFT Peak (0~50Hz)
        'PC1_BandEnergy_Low', # 0~10Hz Energy
        'PC1_BandEnergy_High',# 10~50Hz Energy
        'PC1_SpectralEntropy',# Spectrum complexity
        'PC1_RMSF',
        'PC1_VarianceRatio',
    ],
    'gyro': ['STD_X', 'STD_Y', 'STD_Z'],
    'ir_counter': ['AvgCycleTime']
}
FEATURE_CONFIG_V18 = FEATURE_CONFIG_V19 # Alias for compatibility


WINDOW_SIZE = 1.0  # 1.0 Sec for FFT (1Hz freq resolution)
WINDOW_OVERLAP = 0.5  # 50% overlap

OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ... (Previous Read Functions: Same) ...

def read_influxdb_csv(file_path: str) -> Tuple[pd.DataFrame, float]:
    """InfluxDB CSV 파일 읽기"""
    # Optimized for reading large files with variable structure
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read header only
        first_lines = [next(f) for _ in range(50)]
        f.seek(0)
        lines = f.readlines()
        
    header_lines = [i for i, line in enumerate(lines) if line.startswith('#group')]
    
    if len(header_lines) > 1:
        all_dfs = []
        for idx, header_start in enumerate(header_lines):
            section_end = header_lines[idx + 1] if idx + 1 < len(header_lines) else len(lines)
            from io import StringIO
            section_text = ''.join(lines[header_start:section_end])
            try:
                df = pd.read_csv(StringIO(section_text), skiprows=3, 
                                        dtype={'_time': str, '_value': float, '_field': str})
                all_dfs.append(df)
            except Exception:
                continue
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            return pd.DataFrame(), 1.0
    else:
        df = pd.read_csv(file_path, skiprows=3, dtype={'_time': str, '_field': str}, low_memory=False)
    
    df = df[df['_time'].notna() & (df['_time'] != '_time')]
    df['_value'] = pd.to_numeric(df['_value'], errors='coerce')
    df = df.dropna(subset=['_value'])
    
    df_pivot = df.pivot_table(index='_time', columns='_field', values='_value', aggfunc='first').reset_index()
    df_pivot['_time'] = pd.to_datetime(df_pivot['_time'], utc=True)
    df_pivot = df_pivot.sort_values('_time').reset_index(drop=True)
    df_pivot['Time(s)'] = (df_pivot['_time'] - df_pivot['_time'].iloc[0]).dt.total_seconds()
    
    total_time = df_pivot['Time(s)'].iloc[-1]
    sr = (len(df_pivot) - 1) / total_time if total_time > 0 else 1.0
    return df_pivot, sr

# =====================================
# 연산 함수 (New FFT logic)
# =====================================

def compute_pca(data_3axis: np.ndarray) -> Dict:
    if len(data_3axis) < 3:
        return {'pc1': np.zeros(len(data_3axis)), 'variance_ratio': 0.0}
    mean = np.mean(data_3axis, axis=0)
    centered = data_3axis - mean
    cov = np.cov(centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = eig_vals.argsort()[::-1]
    pc1 = centered @ eig_vecs[:, idx[0]]
    ratio = eig_vals[idx[0]] / np.sum(eig_vals) if np.sum(eig_vals) > 0 else 0
    return {'pc1': pc1, 'variance_ratio': ratio}

def compute_spectral_features(pc1_data: np.ndarray, sr: float) -> Dict:
    """Compute FFT features"""
    N = len(pc1_data)
    if N < 2:
        return {'dom_freq': 0, 'energy_low': 0, 'energy_high': 0, 'entropy': 0, 'rmsf': 0}
        
    # Remove DC (Mean)
    signal = pc1_data - np.mean(pc1_data)
    yf = rfft(signal)
    xf = rfftfreq(N, 1/sr)
    
    magnitude = np.abs(yf)
    power = magnitude ** 2
    
    # 1. Dominant Frequency
    dom_idx = np.argmax(magnitude)
    dom_freq = xf[dom_idx]
    
    # 2. Band Energy (Low: <10Hz, High: >=10Hz)
    # 10Hz is typical boundary for structural sway vs mechanical vibration
    idx_10hz = np.searchsorted(xf, 10.0)
    
    energy_low = np.sum(power[:idx_10hz])
    energy_high = np.sum(power[idx_10hz:])
    
    # Normalize energies by total power to make it scale-invariant? 
    # Or keep absolute for intensity? User wants to detect anomalies -> Absolute is better.
    # But distance changes might affect absolute. Normalized ratio is safer for "pattern".
    # Let's use Log Energy for scale robustness
    log_energy_low = np.log1p(energy_low)
    log_energy_high = np.log1p(energy_high)
    
    # 3. Spectral Entropy
    # Normalize power to probability distribution
    psd_norm = power / np.sum(power)
    spec_entropy = entropy(psd_norm)
    
    # 4. RMSF
    numerator = np.sum((xf ** 2) * power)
    denominator = np.sum(power)
    rmsf = np.sqrt(numerator / denominator) if denominator > 0 else 0.0
    
    return {
        'dom_freq': dom_freq,
        'energy_low': log_energy_low,
        'energy_high': log_energy_high,
        'entropy': spec_entropy,
        'rmsf': rmsf
    }

def extract_features(data_dict: Dict[str, np.ndarray], sr: float) -> Dict[str, float]:
    features = {}
    
    # Accel
    if 'accel' in data_dict:
        acc = data_dict['accel']
        if len(acc) > 0:
            features['accel_VectorRMS'] = np.sqrt(np.mean(np.sum(acc**2, axis=1)))
            pca = compute_pca(acc)
            pc1 = pca['pc1']
            features['accel_PC1_VarianceRatio'] = pca['variance_ratio']
            features['accel_PC1_PeakToPeak'] = np.percentile(pc1, 95) - np.percentile(pc1, 5)
            
            spec = compute_spectral_features(pc1, sr)
            features['accel_PC1_DominantFreq'] = spec['dom_freq']
            features['accel_PC1_BandEnergy_Low'] = spec['energy_low']
            features['accel_PC1_BandEnergy_High'] = spec['energy_high']
            features['accel_PC1_SpectralEntropy'] = spec['entropy']
            features['accel_PC1_RMSF'] = spec['rmsf']
            
    # Gyro
    if 'gyro' in data_dict:
        gyr = data_dict['gyro']
        features['gyro_STD_X'] = np.std(gyr[:, 0])
        features['gyro_STD_Y'] = np.std(gyr[:, 1])
        features['gyro_STD_Z'] = np.std(gyr[:, 2])
        
    # IR
    if 'ir_avg' in data_dict:
        features['ir_AvgCycleTime'] = np.mean(data_dict['ir_avg'])
        
    return features


# Compatibility wrapper for multi_sensor
def process_multi_sensor_files(file_dict: Dict[str, str], resample_rate='10ms', window_size=1.0, window_overlap=0.5):
    # This function is used by regeneration script.
    # Note: resample_rate '10ms' = 100Hz
    # Implement simplified flow here or reuse main logic.
    # For now, just a placeholder as we are rewriting the main file.
    pass

if __name__ == "__main__":
    pass