#!/usr/bin/env python3
"""
Feature Extraction from Sensor Data (InfluxDB Format)
Version 19 (Mean XYZ Added):
- Added: Accel Mean X, Y, Z for tilt/orientation detection.
"""

import os
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =====================================
# 설정
# =====================================

# 센서별 추출할 특징 설정 (V19)
FEATURE_CONFIG_V19 = {
    'accel': [
        'VectorRMS',
        'PC1_PeakToPeak',
        'PC1_DominantFreq',
        'PC1_RMSF',
        'PC1_VarianceRatio',
        'Mean_X', # New
        'Mean_Y', # New
        'Mean_Z', # New
    ],
    'gyro': [
        'STD_X',
        'STD_Y',
        'STD_Z'
    ],
    'ir_counter': [
        'AvgCycleTime',
        'CycleJitter'
    ]
}

# 윈도우 설정
WINDOW_SIZE = 10.0  # 초 단위
WINDOW_OVERLAP = 5.0  # 초 단위

# 출력 디렉토리
OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# CSV 읽기 함수
# =====================================

def read_influxdb_csv(file_path: str) -> Tuple[pd.DataFrame, float]:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header_lines = [i for i, line in enumerate(lines) if line.startswith('#group')]
    
    if len(header_lines) > 1:
        all_dfs = []
        for idx, header_start in enumerate(header_lines):
            if idx + 1 < len(header_lines):
                section_end = header_lines[idx + 1]
            else:
                section_end = len(lines)
            
            from io import StringIO
            section_text = ''.join(lines[header_start:section_end])
            
            try:
                df_section = pd.read_csv(StringIO(section_text), skiprows=3, 
                                        dtype={'_time': str, '_value': float, '_field': str})
                all_dfs.append(df_section)
            except Exception:
                continue
        
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            raise ValueError("Could not read any valid sections")
    else:
        df = pd.read_csv(file_path, skiprows=3, 
                         dtype={'_time': str, '_field': str},
                         low_memory=False)
    
    df = df[df['_time'].notna() & (df['_time'] != '_time')]
    df = df[df['_field'].notna() & (df['_field'] != '_field')]
    df['_value'] = pd.to_numeric(df['_value'], errors='coerce')
    df = df[df['_value'].notna()]
    
    df_pivot = df.pivot_table(
        index='_time',
        columns='_field',
        values='_value',
        aggfunc='first'
    ).reset_index()
    
    df_pivot['_time'] = pd.to_datetime(df_pivot['_time'], format='mixed', utc=True)
    df_pivot = df_pivot.sort_values('_time').reset_index(drop=True)
    df_pivot['Time(s)'] = (df_pivot['_time'] - df_pivot['_time'].iloc[0]).dt.total_seconds()
    
    n_samples = len(df_pivot)
    total_time = df_pivot['Time(s)'].iloc[-1] - df_pivot['Time(s)'].iloc[0]
    sampling_rate = (n_samples - 1) / total_time if total_time > 0 else 1.0
    
    return df_pivot, sampling_rate

# =====================================
# 연산 함수 (PCA, FFT, Stat)
# =====================================

def compute_pca(data_3axis: np.ndarray) -> Dict:
    if len(data_3axis) < 3:
        return {'pc1': np.zeros(len(data_3axis)), 'variance_ratio': 0.0}
    
    mean = np.mean(data_3axis, axis=0)
    centered_data = data_3axis - mean
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    pc1_direction = eigenvectors[:, 0]
    pc1_data = centered_data @ pc1_direction
    
    total_variance = np.sum(eigenvalues)
    variance_ratio = eigenvalues[0] / total_variance if total_variance > 0 else 0.0
    
    return {'pc1': pc1_data, 'variance_ratio': variance_ratio}

def compute_vector_rms(data_3axis: np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum(data_3axis ** 2, axis=1)))

def compute_pc1_peak_to_peak(pc1_data: np.ndarray) -> float:
    return np.ptp(pc1_data)

def compute_pc1_freq_features(pc1_data: np.ndarray, sampling_rate: float) -> Tuple[float, float]:
    N = len(pc1_data)
    if N < 2: return 0.0, 0.0
    
    signal = pc1_data - np.mean(pc1_data)
    spectrum = np.abs(rfft(signal))
    freqs = rfftfreq(N, 1/sampling_rate)
    
    dominant_freq = freqs[np.argmax(spectrum)] if len(spectrum) > 0 else 0.0
    
    power = spectrum ** 2
    numerator = np.sum((freqs ** 2) * power)
    denominator = np.sum(power)
    rmsf = np.sqrt(numerator / denominator) if denominator > 0 else 0.0
    
    return dominant_freq, rmsf

def compute_std_xyz(data_3axis: np.ndarray) -> Tuple[float, float, float]:
    return (np.std(data_3axis[:, 0]), np.std(data_3axis[:, 1]), np.std(data_3axis[:, 2]))

def compute_mean_xyz(data_3axis: np.ndarray) -> Tuple[float, float, float]:
    """각 축 평균 (V19)"""
    return (np.mean(data_3axis[:, 0]), np.mean(data_3axis[:, 1]), np.mean(data_3axis[:, 2]))

# =====================================
# 통합 특징 추출 함수 (V19)
# =====================================

def extract_features(data_dict: Dict[str, np.ndarray], 
                         sampling_rate: float) -> Dict[str, float]:
    """
    특징 추출: 가속도(PCA, Mean), 각속도(STD), IR Counter(Cycle, Jitter)
    """
    features = {}
    
    # 1. 가속도 특징 (8개)
    if 'accel' in data_dict and len(data_dict['accel']) > 0:
        accel_data = data_dict['accel']
        pca_result = compute_pca(accel_data)
        
        features['accel_VectorRMS'] = compute_vector_rms(accel_data)
        features['accel_PC1_PeakToPeak'] = compute_pc1_peak_to_peak(pca_result['pc1'])
        
        dom_freq, rmsf = compute_pc1_freq_features(pca_result['pc1'], sampling_rate)
        features['accel_PC1_DominantFreq'] = dom_freq
        features['accel_PC1_RMSF'] = rmsf
        features['accel_PC1_VarianceRatio'] = pca_result['variance_ratio']
        
        # V19 New Features
        mean_x, mean_y, mean_z = compute_mean_xyz(accel_data)
        features['accel_Mean_X'] = mean_x
        features['accel_Mean_Y'] = mean_y
        features['accel_Mean_Z'] = mean_z
    
    # 2. 각속도 특징 (3개)
    if 'gyro' in data_dict and len(data_dict['gyro']) > 0:
        gyro_data = data_dict['gyro']
        std_x, std_y, std_z = compute_std_xyz(gyro_data)
        features['gyro_STD_X'] = std_x
        features['gyro_STD_Y'] = std_y
        features['gyro_STD_Z'] = std_z
        
    # 3. IR Counter 특징 (2개)
    if 'ir_avg' in data_dict and 'ir_last' in data_dict:
        avg_cycle = data_dict['ir_avg']
        last_cycle = data_dict['ir_last']
        
        if len(avg_cycle) > 0 and len(last_cycle) > 0:
            current_avg_cycle = np.mean(avg_cycle)
            features['ir_AvgCycleTime'] = current_avg_cycle
            
            jitter_values = last_cycle - avg_cycle
            features['ir_CycleJitter'] = np.mean(jitter_values)
    
    return features

# =====================================
# 다중 센서 파일 처리
# =====================================

def process_multi_sensor_files(file_dict: Dict[str, str],
                                    resample_rate: str = '100ms',
                                    window_size: float = WINDOW_SIZE,
                                    window_overlap: float = WINDOW_OVERLAP) -> pd.DataFrame:
    
    print("\n=== Multi-Sensor Processing V19 (Accel/Gyro + IR Counter) ===")
    print(f"Target features: 13 (Accel: 8, Gyro: 3, IR: 2)")
    
    resampled_dfs = []
    
    sensor_fields_map = {
        'accel_gyro': [
            'fields_accel_x', 'fields_accel_y', 'fields_accel_z',
            'fields_gyro_x', 'fields_gyro_y', 'fields_gyro_z',
            'accel_x', 'accel_y', 'accel_z', 
            'gyro_x', 'gyro_y', 'gyro_z'
        ],
        'ir_counter': [
            'avg_cycle_ms', 'last_cycle_ms'
        ]
    }
    
    for sensor_type, file_path in file_dict.items():
        if os.path.exists(file_path):
            try:
                df, sr = read_influxdb_csv(file_path)
                print(f"Loaded {sensor_type}: {len(df)} rows, {sr:.2f}Hz")
                
                df_indexed = df.set_index('_time')
                
                target_fields = sensor_fields_map.get(sensor_type, [])
                available_cols = [c for c in df_indexed.columns if c in target_fields]
                
                if available_cols:
                    df_resampled = df_indexed[available_cols].resample(resample_rate).mean()
                    resampled_dfs.append(df_resampled)
                else:
                    print(f"Warning: No matching fields found in {sensor_type}")
                    
            except Exception as e:
                print(f"Error reading {sensor_type}: {e}")
    
    if not resampled_dfs:
        return pd.DataFrame()
    
    print(f"\nSynchronizing at {resample_rate}...")
    merged_df = resampled_dfs[0]
    for df in resampled_dfs[1:]:
        merged_df = merged_df.join(df, how='outer')
    
    merged_df = merged_df.ffill().bfill()
    merged_df = merged_df.reset_index()
    
    start_time_abs = merged_df['_time'].iloc[0]
    merged_df['Time(s)'] = (merged_df['_time'] - start_time_abs).dt.total_seconds()
    
    print(f"Synchronized Data: {len(merged_df)} samples")
    
    window_step = window_size - window_overlap
    n_samples = len(merged_df)
    total_time = merged_df['Time(s)'].iloc[-1]
    effective_sr = (n_samples - 1) / total_time if total_time > 0 else 1.0
    
    features_list = []
    current_time = 0.0
    window_count = 0
    
    cols = merged_df.columns
    accel_cols = [c for c in cols if 'accel_x' in c] + [c for c in cols if 'accel_y' in c] + [c for c in cols if 'accel_z' in c]
    gyro_cols = [c for c in cols if 'gyro_x' in c] + [c for c in cols if 'gyro_y' in c] + [c for c in cols if 'gyro_z' in c]
    avg_cycle_col = next((c for c in cols if 'avg_cycle_ms' in c), None)
    last_cycle_col = next((c for c in cols if 'last_cycle_ms' in c), None)

    accel_cols.sort()
    gyro_cols.sort()

    while current_time + window_size <= total_time:
        window_end = current_time + window_size
        mask = (merged_df['Time(s)'] >= current_time) & (merged_df['Time(s)'] < window_end)
        window_data = merged_df[mask]
        
        if len(window_data) < 2:
            current_time += window_step
            continue
            
        data_dict = {}
        
        if len(accel_cols) == 3:
            vals = window_data[accel_cols].values
            if not np.isnan(vals).any():
                data_dict['accel'] = vals
        
        if len(gyro_cols) == 3:
            vals = window_data[gyro_cols].values
            if not np.isnan(vals).any():
                data_dict['gyro'] = vals
        
        if avg_cycle_col and last_cycle_col:
            avgs = window_data[avg_cycle_col].values
            lasts = window_data[last_cycle_col].values
            if not np.isnan(avgs).any() and not np.isnan(lasts).any():
                data_dict['ir_avg'] = avgs
                data_dict['ir_last'] = lasts
        
        feats = extract_features(data_dict, effective_sr)
        
        feats['window_id'] = window_count
        feats['start_time'] = current_time
        feats['end_time'] = window_end
        
        features_list.append(feats)
        window_count += 1
        current_time += window_step
        
    print(f"Extracted features from {window_count} windows")
    
    result_df = pd.DataFrame(features_list)
    if not result_df.empty:
        meta_cols = ['window_id', 'start_time', 'end_time']
        feat_cols = [c for c in result_df.columns if c not in meta_cols]
        result_df = result_df[meta_cols + feat_cols]
        
    return result_df

if __name__ == "__main__":
    pass