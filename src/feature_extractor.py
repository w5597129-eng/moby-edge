#!/usr/bin/env python3
"""
Feature Extraction from Sensor Data (InfluxDB Format)
Version 18 (Cycle Time & Jitter Added):
- Removed: Pressure, Temp, Gyro RMS, Direction, Crest Factor
- Added: Avg Cycle Time, Cycle Jitter (from IR Counter)
- Logic: Cycle Jitter = last_cycle_ms - avg_cycle_ms

Target Features: 10 Features
- Accel (5): VectorRMS, PC1_PeakToPeak, PC1_DominantFreq, PC1_RMSF, PC1_VarianceRatio
- Gyro (3): STD_X, STD_Y, STD_Z
- IR (2): AvgCycleTime, CycleJitter

Author: WISE Team, Project MOBY
Date: 2025-11-24 (Modified for V18)
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

# 센서별 추출할 특징 설정 (V18)
FEATURE_CONFIG_V18 = {
    # 3축 가속도: 5개 특징 (핵심 진동/주파수 정보만 유지)
    'accel': [
        'VectorRMS',          # 1. 전체 진동 에너지
        'PC1_PeakToPeak',     # 2. 주축 최대 진폭
        'PC1_DominantFreq',   # 3. 주축 주파수
        'PC1_RMSF',           # 4. 주축 고주파 이동
        'PC1_VarianceRatio',  # 5. 주축 설명력
    ],
    
    # 3축 각속도: 3개 특징 (축별 변동성만 유지)
    'gyro': [
        'STD_X',              # 6. X축 회전 속도 변동성
        'STD_Y',              # 7. Y축 회전 속도 변동성
        'STD_Z'               # 8. Z축 회전 속도 변동성
    ],
    
    # IR 카운터 (신규): 2개 특징
    'ir_counter': [
        'AvgCycleTime',       # 9. 평균 사이클 타임 (ms)
        'CycleJitter'         # 10. 사이클 지터 (ms)
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
    """
    InfluxDB CSV 파일 읽기 (다중 테이블 지원)
    """
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
    
    # Pivot: _field 값을 컬럼으로 변환
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
    """3축 데이터 PCA 수행"""
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
    """Vector RMS: 전체 에너지"""
    return np.sqrt(np.mean(np.sum(data_3axis ** 2, axis=1)))

def compute_pc1_peak_to_peak(pc1_data: np.ndarray) -> float:
    """PC1 Peak-to-Peak"""
    return np.ptp(pc1_data)

def compute_pc1_freq_features(pc1_data: np.ndarray, sampling_rate: float) -> Tuple[float, float]:
    """PC1 주파수 특징 (Dominant Freq, RMSF)"""
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
    """각 축 표준편차"""
    return (np.std(data_3axis[:, 0]), np.std(data_3axis[:, 1]), np.std(data_3axis[:, 2]))

# =====================================
# 통합 특징 추출 함수 (V18 수정됨)
# =====================================

def extract_features(data_dict: Dict[str, np.ndarray], 
                         sampling_rate: float) -> Dict[str, float]:
    """
    특징 추출: 가속도(PCA), 각속도(STD), IR Counter(Cycle, Jitter)
    """
    features = {}
    
    # 1. 가속도 특징 (5개)
    if 'accel' in data_dict and len(data_dict['accel']) > 0:
        accel_data = data_dict['accel']
        pca_result = compute_pca(accel_data)
        
        features['accel_VectorRMS'] = compute_vector_rms(accel_data)
        features['accel_PC1_PeakToPeak'] = compute_pc1_peak_to_peak(pca_result['pc1'])
        
        dom_freq, rmsf = compute_pc1_freq_features(pca_result['pc1'], sampling_rate)
        features['accel_PC1_DominantFreq'] = dom_freq
        features['accel_PC1_RMSF'] = rmsf
        features['accel_PC1_VarianceRatio'] = pca_result['variance_ratio']
    
    # 2. 각속도 특징 (3개)
    if 'gyro' in data_dict and len(data_dict['gyro']) > 0:
        gyro_data = data_dict['gyro']
        std_x, std_y, std_z = compute_std_xyz(gyro_data)
        features['gyro_STD_X'] = std_x
        features['gyro_STD_Y'] = std_y
        features['gyro_STD_Z'] = std_z
        
    # 3. IR Counter 특징 (2개) - 신규 추가
    if 'ir_avg' in data_dict and 'ir_last' in data_dict:
        avg_cycle = data_dict['ir_avg']
        last_cycle = data_dict['ir_last']
        
        if len(avg_cycle) > 0 and len(last_cycle) > 0:
            # 평균 사이클 타임 (윈도우 내 평균)
            current_avg_cycle = np.mean(avg_cycle)
            features['ir_AvgCycleTime'] = current_avg_cycle
            
            # 사이클 지터 = last_cycle_ms - 평균 사이클 타임
            # (윈도우 내에서의 편차 평균)
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
    
    print("\n=== Multi-Sensor Processing V18 (Accel/Gyro + IR Counter) ===")
    print(f"Target features: 10 (Accel: 5, Gyro: 3, IR: 2)")
    
    # 1. 각 센서 파일 읽기 및 필드 매핑
    resampled_dfs = []
    
    # 센서별 필요한 필드 정의 (InfluxDB _field 값 기준)
    # 1124_accel_gyro 파일은 보통 fields_ 접두어가 붙거나, 원본에 따라 다를 수 있음.
    # 1124_IRcounter 파일은 avg_cycle_ms, last_cycle_ms 필드를 가짐
    sensor_fields_map = {
        'accel_gyro': [
            'fields_accel_x', 'fields_accel_y', 'fields_accel_z',
            'fields_gyro_x', 'fields_gyro_y', 'fields_gyro_z',
            'accel_x', 'accel_y', 'accel_z', # 접두어 없는 경우 대비
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
                
                # 사용 가능한 컬럼만 필터링
                target_fields = sensor_fields_map.get(sensor_type, [])
                available_cols = [c for c in df_indexed.columns if c in target_fields]
                
                if available_cols:
                    # 리샘플링 (평균)
                    df_resampled = df_indexed[available_cols].resample(resample_rate).mean()
                    resampled_dfs.append(df_resampled)
                else:
                    print(f"Warning: No matching fields found in {sensor_type}. Columns: {df_indexed.columns.tolist()}")
                    
            except Exception as e:
                print(f"Error reading {sensor_type}: {e}")
    
    if not resampled_dfs:
        return pd.DataFrame()
    
    # 2. 데이터 병합 (Outer Join)
    print(f"\nSynchronizing at {resample_rate}...")
    merged_df = resampled_dfs[0]
    for df in resampled_dfs[1:]:
        merged_df = merged_df.join(df, how='outer')
    
    merged_df = merged_df.ffill().bfill()
    merged_df = merged_df.reset_index()
    
    # 상대 시간 계산 (가동 시작 시각 기준)
    start_time_abs = merged_df['_time'].iloc[0]
    merged_df['Time(s)'] = (merged_df['_time'] - start_time_abs).dt.total_seconds()
    
    print(f"Synchronized Data: {len(merged_df)} samples, Duration: {merged_df['Time(s)'].iloc[-1]:.2f}s")
    
    # 3. 윈도우링 및 특징 추출
    window_step = window_size - window_overlap
    n_samples = len(merged_df)
    total_time = merged_df['Time(s)'].iloc[-1]
    effective_sr = (n_samples - 1) / total_time if total_time > 0 else 1.0
    
    features_list = []
    current_time = 0.0
    window_count = 0
    
    # 컬럼명 정리 (접두어 처리)
    cols = merged_df.columns
    accel_cols = [c for c in cols if 'accel_x' in c] + [c for c in cols if 'accel_y' in c] + [c for c in cols if 'accel_z' in c]
    gyro_cols = [c for c in cols if 'gyro_x' in c] + [c for c in cols if 'gyro_y' in c] + [c for c in cols if 'gyro_z' in c]
    # IR 컬럼 찾기 (정확한 매칭)
    avg_cycle_col = next((c for c in cols if 'avg_cycle_ms' in c), None)
    last_cycle_col = next((c for c in cols if 'last_cycle_ms' in c), None)

    # 3축 순서 보장 (x, y, z)
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
        
        # Accel Data
        if len(accel_cols) == 3:
            vals = window_data[accel_cols].values
            if not np.isnan(vals).any():
                data_dict['accel'] = vals
        
        # Gyro Data
        if len(gyro_cols) == 3:
            vals = window_data[gyro_cols].values
            if not np.isnan(vals).any():
                data_dict['gyro'] = vals
        
        # IR Data
        if avg_cycle_col and last_cycle_col:
            avgs = window_data[avg_cycle_col].values
            lasts = window_data[last_cycle_col].values
            if not np.isnan(avgs).any() and not np.isnan(lasts).any():
                data_dict['ir_avg'] = avgs
                data_dict['ir_last'] = lasts
        
        # 특징 추출
        feats = extract_features(data_dict, effective_sr)
        
        # 메타데이터
        feats['window_id'] = window_count
        feats['start_time'] = current_time
        feats['end_time'] = window_end
        
        features_list.append(feats)
        window_count += 1
        current_time += window_step
        
    print(f"Extracted features from {window_count} windows")
    
    # 결과 생성
    result_df = pd.DataFrame(features_list)
    if not result_df.empty:
        meta_cols = ['window_id', 'start_time', 'end_time']
        feat_cols = [c for c in result_df.columns if c not in meta_cols]
        result_df = result_df[meta_cols + feat_cols]
        
    return result_df

# =====================================
# 메인 실행
# =====================================

if __name__ == "__main__":
    # 파일 경로 설정 (사용자 환경에 맞게 수정 필요)
    input_files = {
        'accel_gyro': 'data/raw/1124 sensor_data/1124_accel_gyro_normal.csv',
        'ir_counter': 'data/raw/1124 sensor_data/1124_IRcounter_normal.csv'
    }
    
    # 파일 존재 확인
    valid_files = {k: v for k, v in input_files.items() if os.path.exists(v)}
    
    if valid_files:
        features_df = process_multi_sensor_files(
            valid_files,
            resample_rate='78.125ms',  # 약 12.8Hz (센서 주기에 맞춤)
            window_size=10.0,
            window_overlap=5.0
        )
        
        if not features_df.empty:
            out_path = os.path.join(OUTPUT_DIR, "1124_features_v18.csv")
            features_df.to_csv(out_path, index=False)
            print(f"\n✓ Saved: {out_path}")
            print(f"✓ Shape: {features_df.shape}")
            print("✓ Features extracted:")
            for col in features_df.columns[3:]:
                print(f"  - {col}")
    else:
        print("No valid files found. Check paths.")