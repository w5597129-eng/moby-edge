#!/usr/bin/env python3
"""
Feature Extraction from Sensor Data (InfluxDB Format)
Version 17 (PCA + Vector Scalarization + FFT Reuse):
- Vector scalarization: 3-axis → scalar features
- PCA-based dimensionality reduction
- FFT reuse optimization
- Maximum computational efficiency

Key Improvements over V16:
- Feature count: 23 → 15 (34.8% reduction)
- FFT calls: 6 → 1 (83.3% reduction!) ⭐
- Speed: ~60-65% faster (FFT-weighted)

Optimizations:
- PCA: 3 axes → 1 principal component (67% FFT reduction)
- FFT reuse: Dominant Freq + RMSF share same spectrum (50% additional)
- Vector RMS: Eliminated redundant sqrt operations

Version 16 (Optimized):
- Sensor-specific feature extraction (only extract what's needed)
- Added RMS Frequency (RMSF) feature
- Reduced feature count from 77(11 per field) to 23

Author: WISE Team, Project MOBY
Date: 2025-11-21 (Optimized)
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

# 센서별 추출할 특징 설정 (V17)
FEATURE_CONFIG_V17 = {
    # 3축 가속도: 9개 특징 (벡터 스칼라화 + PCA)
    'accel': [
        'VectorRMS',          # 1. 전체 진동 에너지
        'PC1_PeakToPeak',     # 2. 주축 최대 진폭
        'VectorCrestFactor',  # 3. 충격도 (vector norm 기반)
        'PC1_DominantFreq',   # 4. 주축 주파수
        'PC1_RMSF',           # 5. 주축 고주파 이동
        'PC1_VarianceRatio',  # 6. 주축 설명력 (단일축 지배도)
        'PC1_Direction_X',    # 7. 주축 방향 X 성분
        'PC1_Direction_Y',    # 8. 주축 방향 Y 성분
        'PC1_Direction_Z'     # 9. 주축 방향 Z 성분
    ],
    
    # 3축 각속도: 4개 특징 (벡터 스칼라화 + 축별 변동성)
    'gyro': [
        'VectorRMS',          # 1. 속도 불안정성 총량
        'STD_X',              # 2. X축 회전 속도 변동성
        'STD_Y',              # 3. Y축 회전 속도 변동성
        'STD_Z'               # 4. Z축 회전 속도 변동성
    ],
    
    # 환경 센서: 2개 특징
    'pressure': ['Mean'],
    'temperature': ['Mean']
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
    InfluxDB CSV 파일 읽기 (V16과 동일)
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
            except Exception as e:
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
# PCA 함수
# =====================================

def compute_pca(data_3axis: np.ndarray) -> Dict:
    """
    3축 데이터에 대한 PCA 수행
    
    Parameters:
    - data_3axis: (n_samples, 3) shape의 numpy array
    
    Returns:
    - dict with keys:
        - 'pc1': First principal component (n_samples,)
        - 'variance_ratio': Explained variance ratio of PC1
        - 'direction': PC1 direction vector (3,)
        - 'centered_data': Centered data (n_samples, 3)
    """
    if len(data_3axis) < 3:
        return {
            'pc1': np.zeros(len(data_3axis)),
            'variance_ratio': 0.0,
            'direction': np.array([0.0, 0.0, 0.0]),
            'centered_data': data_3axis
        }
    
    # 평균 제거 (centering)
    mean = np.mean(data_3axis, axis=0)
    centered_data = data_3axis - mean
    
    # 공분산 행렬
    cov_matrix = np.cov(centered_data, rowvar=False)
    
    # 고유값, 고유벡터 계산
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 고유값 내림차순 정렬
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # PC1 (첫 번째 주성분)
    pc1_direction = eigenvectors[:, 0]
    pc1_data = centered_data @ pc1_direction
    
    # PC1 설명력 (분산 비율)
    total_variance = np.sum(eigenvalues)
    variance_ratio = eigenvalues[0] / total_variance if total_variance > 0 else 0.0
    
    return {
        'pc1': pc1_data,
        'variance_ratio': variance_ratio,
        'direction': pc1_direction,
        'centered_data': centered_data
    }

# =====================================
# 벡터 특징 함수들
# =====================================

def compute_vector_rms(data_3axis: np.ndarray) -> float:
    """
    Vector RMS: sqrt(mean(||v||²))
    
    Optimized: Eliminates unnecessary sqrt → square cancellation
    
    물리적 의미: 3축의 총 진동 에너지 (방향 무관)
    """
    return np.sqrt(np.mean(np.sum(data_3axis ** 2, axis=1)))

def compute_vector_crest_factor(data_3axis: np.ndarray) -> float:
    """
    Vector Crest Factor: max(||v||) / RMS(||v||)
    
    물리적 의미: 벡터 크기의 충격도 (축 독립적)
    """
    vector_magnitude = np.sqrt(np.sum(data_3axis ** 2, axis=1))
    peak = np.max(vector_magnitude)
    rms = np.sqrt(np.mean(vector_magnitude ** 2))
    return peak / rms if rms > 0 else 0.0

def compute_pc1_peak_to_peak(pc1_data: np.ndarray) -> float:
    """PC1 축의 Peak-to-Peak"""
    return np.ptp(pc1_data)

def compute_pc1_freq_features(pc1_data: np.ndarray, sampling_rate: float) -> Tuple[float, float]:
    """
    PC1 축의 주파수 특징 (Dominant Freq + RMSF)
    
    FFT를 한 번만 수행하고 두 특징을 모두 계산
    
    Parameters:
    - pc1_data: PC1 시계열 데이터
    - sampling_rate: 샘플링 주파수
    
    Returns:
    - (dominant_freq, rmsf) tuple
    
    Optimization: 기존에는 FFT를 2번 호출했지만,
                  같은 스펙트럼을 공유하므로 1번만 호출!
    """
    N = len(pc1_data)
    if N < 2:
        return 0.0, 0.0
    
    # DC 제거
    signal = pc1_data - np.mean(pc1_data)
    
    # === FFT 한 번만! ===
    spectrum = np.abs(rfft(signal))
    freqs = rfftfreq(N, 1/sampling_rate)
    
    # === Dominant Frequency (스펙트럼 재사용) ===
    dominant_freq = freqs[np.argmax(spectrum)] if len(spectrum) > 0 else 0.0
    
    # === RMSF (스펙트럼 재사용) ===
    power = spectrum ** 2
    numerator = np.sum((freqs ** 2) * power)
    denominator = np.sum(power)
    rmsf = np.sqrt(numerator / denominator) if denominator > 0 else 0.0
    
    return dominant_freq, rmsf

def compute_mean(signal: np.ndarray) -> float:
    """Mean"""
    return np.mean(signal)

def compute_std_xyz(data_3axis: np.ndarray) -> Tuple[float, float, float]:
    """
    각 축의 표준편차 (Standard Deviation)
    
    물리적 의미: 각 축별 회전 속도 변동성
    
    Parameters:
    - data_3axis: (n_samples, 3) numpy array
    
    Returns:
    - (std_x, std_y, std_z) tuple
    """
    return (
        np.std(data_3axis[:, 0]),
        np.std(data_3axis[:, 1]),
        np.std(data_3axis[:, 2])
    )

# =====================================
# 통합 특징 추출 함수
# =====================================

def extract_features_v17(data_dict: Dict[str, np.ndarray], 
                         sampling_rate: float) -> Dict[str, float]:
    """
    V17 특징 추출: PCA + 벡터 스칼라화
    
    Parameters:
    - data_dict: {
        'accel': (n, 3) numpy array,
        'gyro': (n, 3) numpy array,
        'pressure': (n,) numpy array,
        'temperature': (n,) numpy array
      }
    - sampling_rate: 샘플링 주파수
    
    Returns:
    - features: {feature_name: value} 딕셔너리
    """
    features = {}
    
    # ===== 가속도 특징 (9개) =====
    if 'accel' in data_dict and len(data_dict['accel']) > 0:
        accel_data = data_dict['accel']
        
        # PCA 수행
        pca_result = compute_pca(accel_data)
        
        # 1. Vector RMS
        features['accel_VectorRMS'] = compute_vector_rms(accel_data)
        
        # 2. PC1 Peak-to-Peak
        features['accel_PC1_PeakToPeak'] = compute_pc1_peak_to_peak(pca_result['pc1'])
        
        # 3. Vector Crest Factor
        features['accel_VectorCrestFactor'] = compute_vector_crest_factor(accel_data)
        
        # 4-5. PC1 Dominant Frequency + RMSF (FFT 한 번에 둘 다!)
        dominant_freq, rmsf = compute_pc1_freq_features(pca_result['pc1'], sampling_rate)
        features['accel_PC1_DominantFreq'] = dominant_freq
        features['accel_PC1_RMSF'] = rmsf
        
        # 6. PC1 Variance Ratio (설명력)
        features['accel_PC1_VarianceRatio'] = pca_result['variance_ratio']
        
        # 7-9. PC1 Direction (방향 벡터)
        features['accel_PC1_Direction_X'] = pca_result['direction'][0]
        features['accel_PC1_Direction_Y'] = pca_result['direction'][1]
        features['accel_PC1_Direction_Z'] = pca_result['direction'][2]
    
    # ===== 각속도 특징 (4개) =====
    if 'gyro' in data_dict and len(data_dict['gyro']) > 0:
        gyro_data = data_dict['gyro']
        
        # 1. Vector RMS (총 불안정성)
        features['gyro_VectorRMS'] = compute_vector_rms(gyro_data)
        
        # 2-4. 축별 표준편차 (방향별 변동성)
        std_x, std_y, std_z = compute_std_xyz(gyro_data)
        features['gyro_STD_X'] = std_x
        features['gyro_STD_Y'] = std_y
        features['gyro_STD_Z'] = std_z
    
    # ===== 환경 특징 (2개) =====
    if 'pressure' in data_dict and len(data_dict['pressure']) > 0:
        features['pressure_Mean'] = compute_mean(data_dict['pressure'])
    
    if 'temperature' in data_dict and len(data_dict['temperature']) > 0:
        features['temperature_Mean'] = compute_mean(data_dict['temperature'])
    
    return features

# =====================================
# 다중 센서 파일 처리
# =====================================

def process_multi_sensor_files_v17(file_dict: Dict[str, str],
                                    resample_rate: str = '100ms',
                                    window_size: float = WINDOW_SIZE,
                                    window_overlap: float = WINDOW_OVERLAP) -> pd.DataFrame:
    """
    여러 센서 파일을 동기화하여 V17 특징 추출
    
    Parameters:
    - file_dict: {sensor_type: file_path} 딕셔너리
    - resample_rate: 동기화 시 리샘플링 주기
    - window_size: 윈도우 크기 (초)
    - window_overlap: 윈도우 겹침 (초)
    
    Returns:
    - 특징 DataFrame
    """
    
    print("\n=== Multi-Sensor Processing V17 (PCA + Vector Scalarization) ===")
    print(f"Expected features: 15 (Accel: 9, Gyro: 4, Env: 2)")
    
    # 1. 각 센서 파일 독립적으로 읽고 리샘플링
    resampled_dfs = []
    sensor_info = []
    
    sensor_fields = {
        'accel_gyro': ['fields_accel_x', 'fields_accel_y', 'fields_accel_z',
                       'fields_gyro_x', 'fields_gyro_y', 'fields_gyro_z'],
        'pressure': ['fields_pressure_hpa', 'fields_temperature_c']
    }
    
    for sensor_type, file_path in file_dict.items():
        if os.path.exists(file_path):
            try:
                df, sr = read_influxdb_csv(file_path)
                sensor_info.append(f"{sensor_type}: {len(df)} samples @ {sr:.2f} Hz")
                
                df_indexed = df.set_index('_time')
                
                # 해당 센서의 필드들 선택
                available_fields = [col for col in df_indexed.columns 
                                   if col in sensor_fields.get(sensor_type, [])]
                
                if available_fields:
                    df_resampled = df_indexed[available_fields].resample(resample_rate).mean()
                    resampled_dfs.append(df_resampled)
                    
            except Exception as e:
                print(f"  Error reading {sensor_type}: {e}")
    
    for info in sensor_info:
        print(f"  {info}")
    
    if not resampled_dfs:
        return pd.DataFrame()
    
    # 2. Outer join으로 병합
    print(f"\nSynchronizing at {resample_rate}...")
    
    merged_df = resampled_dfs[0].copy()
    for df in resampled_dfs[1:]:
        merged_df = merged_df.join(df, how='outer')
    
    # 3. NaN 보간
    merged_df = merged_df.ffill().bfill()
    
    # 4. 상대 시간 추가
    merged_df = merged_df.reset_index()
    merged_df['Time(s)'] = (merged_df['_time'] - merged_df['_time'].iloc[0]).dt.total_seconds()
    
    print(f"Synchronized: {len(merged_df)} samples")
    
    # 5. 윈도우 기반 특징 추출
    window_step = window_size - window_overlap
    
    n_samples = len(merged_df)
    total_time = merged_df['Time(s)'].iloc[-1] - merged_df['Time(s)'].iloc[0]
    effective_sr = (n_samples - 1) / total_time if total_time > 0 else 1.0
    
    print(f"\nExtracting V17 features with {window_size}s windows (overlap: {window_overlap}s)...")
    print(f"Effective sampling rate: {effective_sr:.2f} Hz")
    
    features_list = []
    
    start_time = merged_df['Time(s)'].iloc[0]
    end_time = merged_df['Time(s)'].iloc[-1]
    
    current_time = start_time
    window_count = 0
    
    while current_time + window_size <= end_time:
        window_end = current_time + window_size
        
        # 윈도우 데이터 추출
        window_mask = (merged_df['Time(s)'] >= current_time) & (merged_df['Time(s)'] < window_end)
        window_data = merged_df[window_mask]
        
        if len(window_data) < 2:
            current_time += window_step
            continue
        
        # 데이터 준비
        data_dict = {}
        
        # 가속도 3축
        accel_cols = ['fields_accel_x', 'fields_accel_y', 'fields_accel_z']
        if all(col in window_data.columns for col in accel_cols):
            accel_3axis = window_data[accel_cols].values
            # NaN 제거
            valid_mask = ~np.isnan(accel_3axis).any(axis=1)
            if valid_mask.sum() > 0:
                data_dict['accel'] = accel_3axis[valid_mask]
        
        # 각속도 3축
        gyro_cols = ['fields_gyro_x', 'fields_gyro_y', 'fields_gyro_z']
        if all(col in window_data.columns for col in gyro_cols):
            gyro_3axis = window_data[gyro_cols].values
            valid_mask = ~np.isnan(gyro_3axis).any(axis=1)
            if valid_mask.sum() > 0:
                data_dict['gyro'] = gyro_3axis[valid_mask]
        
        # 환경
        if 'fields_pressure_hpa' in window_data.columns:
            pressure = window_data['fields_pressure_hpa'].values
            pressure = pressure[~np.isnan(pressure)]
            if len(pressure) > 0:
                data_dict['pressure'] = pressure
        
        if 'fields_temperature_c' in window_data.columns:
            temperature = window_data['fields_temperature_c'].values
            temperature = temperature[~np.isnan(temperature)]
            if len(temperature) > 0:
                data_dict['temperature'] = temperature
        
        # 특징 추출
        features = extract_features_v17(data_dict, effective_sr)
        
        # 메타데이터 추가
        features['window_id'] = window_count
        features['start_time'] = current_time
        features['end_time'] = window_end
        
        features_list.append(features)
        window_count += 1
        current_time += window_step
    
    print(f"Extracted features from {window_count} windows")
    
    result_df = pd.DataFrame(features_list)
    
    # 컬럼 순서 정리 (메타데이터 먼저)
    meta_cols = ['window_id', 'start_time', 'end_time']
    feature_cols = [col for col in result_df.columns if col not in meta_cols]
    result_df = result_df[meta_cols + feature_cols]
    
    print(f"Total feature columns: {len(feature_cols)}")
    
    return result_df

# =====================================
# 메인 실행
# =====================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Feature Extraction V17 - PCA + Vector Scalarization")
    print("=" * 70)
    
    sensor_files = {
        'accel_gyro': 'data/raw/1120 sensor_data/1120_accel_gyro_normal.csv',
        'pressure': 'data/raw/1120 sensor_data/1120_pressure_normal.csv',
    }
    
    valid_files = {k: v for k, v in sensor_files.items() if os.path.exists(v)}
    
    if valid_files:
        features = process_multi_sensor_files_v17(
            valid_files,
            resample_rate='78.125ms',  # ~12.8Hz
            window_size=WINDOW_SIZE,
            window_overlap=WINDOW_OVERLAP
        )
        
        if not features.empty:
            output_path = os.path.join(OUTPUT_DIR, "1120_features.csv")
            features.to_csv(output_path, index=False)
            print(f"\n✓ Saved to: {output_path}")
            print(f"✓ Shape: {features.shape}")
            print(f"✓ Features per window: {features.shape[1] - 3}")
    else:
        print("No valid files found. Please check file paths.")