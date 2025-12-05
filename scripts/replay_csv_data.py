#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 데이터 리플레이 스크립트

수집된 InfluxDB CSV 데이터를 읽어 실시간처럼 MQTT로 윈도우 메시지를 발행합니다.
inference_worker.py가 이 메시지를 구독하여 추론을 수행합니다.

사용법:
    python scripts/replay_csv_data.py --csv data/1205_accel_gyro_s1_red.csv --speed 1.0

옵션:
    --csv: CSV 파일 경로 (필수)
    --ir-csv: IR 센서 CSV 파일 경로 (옵션, accel_gyro와 함께 사용)
    --speed: 재생 속도 배율 (기본: 1.0, 2.0이면 2배속)
    --broker: MQTT 브로커 주소 (기본: 192.168.80.208)
    --port: MQTT 포트 (기본: 1883)
    --window-size: 윈도우 크기(초) (기본: 10.0)
    --window-overlap: 윈도우 오버랩(초) (기본: 5.0)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference_interface import (
    WINDOW_SIZE,
    WINDOW_OVERLAP,
    WINDOW_TOPIC_ROOT,
    WindowMessage,
    current_timestamp_ns,
)


def parse_influxdb_csv(csv_path: str) -> pd.DataFrame:
    """
    InfluxDB 내보내기 CSV를 파싱합니다.
    
    InfluxDB CSV 형식:
    - 첫 3줄은 메타데이터 (#group, #datatype, #default)
    - 4번째 줄은 헤더
    - 5번째 줄부터 데이터
    - _time, _value, _field 컬럼 사용
    """
    # 메타데이터 줄 건너뛰기
    df = pd.read_csv(csv_path, skiprows=3)
    
    # 필요한 컬럼 확인
    required_cols = ['_time', '_value', '_field']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # 시간 파싱
    df['_time'] = pd.to_datetime(df['_time'])
    
    return df


def pivot_to_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    InfluxDB 긴 형식(long format)을 넓은 형식(wide format)으로 변환합니다.
    
    입력: _time, _value, _field 형태의 긴 형식
    출력: 각 필드가 컬럼이 되고 시간이 인덱스인 넓은 형식
    """
    # pivot: _field 값들이 컬럼이 됨
    pivoted = df.pivot_table(
        index='_time',
        columns='_field',
        values='_value',
        aggfunc='first'  # 동일 시간에 중복이 있으면 첫 번째 사용
    ).reset_index()
    
    # 시간순 정렬
    pivoted = pivoted.sort_values('_time').reset_index(drop=True)
    
    return pivoted


def create_mqtt_client(broker: str, port: int) -> mqtt.Client:
    """MQTT 클라이언트 생성 및 연결"""
    try:
        client = mqtt.Client(client_id="csv_replay_publisher")
    except TypeError:
        # paho-mqtt v2 호환
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="csv_replay_publisher")
    
    try:
        client.connect(broker, port, 60)
        client.loop_start()
        print(f"✅ MQTT 브로커 연결됨: {broker}:{port}")
    except Exception as e:
        print(f"❌ MQTT 연결 실패: {e}")
        sys.exit(1)
    
    return client


def detect_sensor_type(df: pd.DataFrame) -> str:
    """CSV 데이터에서 센서 타입 자동 감지"""
    columns = set(df.columns)
    
    accel_cols = {'fields_accel_x', 'fields_accel_y', 'fields_accel_z'}
    gyro_cols = {'fields_gyro_x', 'fields_gyro_y', 'fields_gyro_z'}
    ir_cols = {'avg_cycle_ms', 'last_cycle_ms'}
    
    if accel_cols.issubset(columns) or gyro_cols.issubset(columns):
        return "accel_gyro"
    elif ir_cols.issubset(columns):
        return "ir_counter"
    else:
        return "unknown"


def build_window_message(
    sensor_type: str,
    window_df: pd.DataFrame,
    sampling_rate_hz: float
) -> Optional[WindowMessage]:
    """윈도우 데이터프레임에서 WindowMessage 생성"""
    
    if len(window_df) < 2:
        return None
    
    window_fields = {}
    
    # 각 필드별로 데이터 추출
    for col in window_df.columns:
        if col == '_time':
            # 타임스탬프는 나노초로 변환
            timestamps = window_df['_time'].astype(np.int64).tolist()
            window_fields['timestamp_ns'] = timestamps
        elif col.startswith('fields_') or col in ['avg_cycle_ms', 'last_cycle_ms']:
            # 필드 데이터
            values = window_df[col].dropna().tolist()
            if values:
                # IR 센서 필드명 매핑
                if col == 'avg_cycle_ms':
                    window_fields['fields_avg_cycle_ms'] = values
                elif col == 'last_cycle_ms':
                    window_fields['fields_last_cycle_ms'] = values
                else:
                    window_fields[col] = values
    
    # 데이터가 없으면 None 반환
    if not any(k for k in window_fields.keys() if k != 'timestamp_ns'):
        return None
    
    return WindowMessage(
        sensor_type=sensor_type,
        sampling_rate_hz=sampling_rate_hz,
        window_fields=window_fields,
        timestamp_ns=current_timestamp_ns(),
    )


def replay_csv(
    csv_path: str,
    ir_csv_path: Optional[str],
    broker: str,
    port: int,
    speed: float,
    window_size: float,
    window_overlap: float,
    interval: float = 5.0,
):
    """CSV 데이터를 실시간처럼 재생"""
    
    print(f"\n{'='*70}")
    print(f"{'CSV DATA REPLAY':^70}")
    print(f"{'='*70}")
    print(f"CSV 파일: {csv_path}")
    if ir_csv_path:
        print(f"IR CSV 파일: {ir_csv_path}")
    print(f"발행 간격: {interval}초")
    print(f"윈도우 크기: {window_size}초, 오버랩: {window_overlap}초")
    print(f"{'='*70}\n")
    
    # CSV 파싱
    print("📂 CSV 데이터 로드 중...")
    raw_df = parse_influxdb_csv(csv_path)
    df = pivot_to_timeseries(raw_df)
    
    # IR 데이터 병합 (옵션)
    if ir_csv_path and os.path.exists(ir_csv_path):
        print(f"📂 IR CSV 데이터 로드 중...")
        ir_raw_df = parse_influxdb_csv(ir_csv_path)
        ir_df = pivot_to_timeseries(ir_raw_df)
        # 시간 기반으로 병합
        df = pd.merge_asof(
            df.sort_values('_time'),
            ir_df.sort_values('_time'),
            on='_time',
            direction='nearest',
            tolerance=pd.Timedelta('1s')
        )
    
    # 센서 타입 감지
    sensor_type = detect_sensor_type(df)
    print(f"📊 감지된 센서 타입: {sensor_type}")
    print(f"📊 총 샘플 수: {len(df)}")
    
    # 시간 범위
    time_range = (df['_time'].max() - df['_time'].min()).total_seconds()
    print(f"📊 데이터 시간 범위: {time_range:.1f}초")
    
    # 샘플링 레이트 계산
    time_diffs = df['_time'].diff().dropna()
    avg_interval = time_diffs.mean().total_seconds()
    sampling_rate_hz = 1.0 / avg_interval if avg_interval > 0 else 12.8
    print(f"📊 추정 샘플링 레이트: {sampling_rate_hz:.2f} Hz")
    
    # 컬럼 정보
    data_cols = [c for c in df.columns if c != '_time']
    print(f"📊 데이터 필드: {', '.join(data_cols[:5])}{'...' if len(data_cols) > 5 else ''}")
    
    # MQTT 연결
    client = create_mqtt_client(broker, port)
    
    # 윈도우 설정
    window_step = window_size - window_overlap
    topic = f"{WINDOW_TOPIC_ROOT}/{sensor_type}"
    
    print(f"\n🚀 재생 시작! (Ctrl+C로 중단)")
    print(f"📤 MQTT 토픽: {topic}")
    print(f"{'='*70}\n")
    
    start_time = df['_time'].min()
    end_time = df['_time'].max()
    
    current_window_start = start_time
    window_count = 0
    real_start = time.time()
    
    try:
        while current_window_start < end_time:
            # 윈도우 데이터 추출
            window_end = current_window_start + pd.Timedelta(seconds=window_size)
            window_df = df[(df['_time'] >= current_window_start) & (df['_time'] < window_end)]
            
            if len(window_df) > 0:
                # 윈도우 메시지 생성
                window_msg = build_window_message(sensor_type, window_df, sampling_rate_hz)
                
                if window_msg:
                    window_count += 1
                    payload = window_msg.to_payload()
                    
                    # MQTT 발행
                    result = client.publish(topic, json.dumps(payload))
                    
                    # 출력
                    elapsed_data_time = (current_window_start - start_time).total_seconds()
                    sample_count = sum(
                        len(v) for k, v in window_msg.window_fields.items() 
                        if k != 'timestamp_ns' and isinstance(v, list)
                    )
                    
                    print(f"[Window #{window_count:03d}] "
                          f"Data Time: {elapsed_data_time:6.1f}s | "
                          f"Samples: {sample_count:4d} | "
                          f"Published: {'✅' if result.rc == 0 else '❌'}")
            
            # 다음 윈도우로 이동
            current_window_start += pd.Timedelta(seconds=window_step)
            
            # 고정 간격으로 발행 (5초 기본)
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  재생 중단됨")
    
    finally:
        elapsed_real = time.time() - real_start
        print(f"\n{'='*70}")
        print(f"📊 재생 완료 통계")
        print(f"{'='*70}")
        print(f"  - 총 윈도우 수: {window_count}")
        print(f"  - 데이터 시간: {time_range:.1f}초")
        print(f"  - 실제 소요 시간: {elapsed_real:.1f}초")
        print(f"  - 유효 속도: {time_range/elapsed_real:.1f}x")
        print(f"{'='*70}\n")
        
        client.loop_stop()
        client.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="수집된 CSV 데이터를 실시간처럼 MQTT로 재생합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    # 단일 CSV 재생 (1배속)
    python scripts/replay_csv_data.py --csv data/1205_accel_gyro_s1_red.csv
    
    # 2배속 재생
    python scripts/replay_csv_data.py --csv data/1205_accel_gyro_normal.csv --speed 2
    
    # Accel/Gyro + IR 데이터 함께 재생
    python scripts/replay_csv_data.py --csv data/1205_accel_gyro_s1_red.csv \\
                                       --ir-csv data/1205_IRcounter_s1_red.csv
    
    # 빠른 테스트 (10배속, 대기 없음)
    python scripts/replay_csv_data.py --csv data/1205_accel_gyro_s1_red.csv --speed 10
        """
    )
    
    parser.add_argument('--csv', required=True, help='메인 CSV 파일 경로')
    parser.add_argument('--ir-csv', help='IR 센서 CSV 파일 경로 (옵션)')
    parser.add_argument('--speed', type=float, default=1.0, help='재생 속도 배율 (기본: 1.0)')
    parser.add_argument('--broker', default='192.168.80.208', help='MQTT 브로커 주소')
    parser.add_argument('--port', type=int, default=1883, help='MQTT 포트')
    parser.add_argument('--window-size', type=float, default=WINDOW_SIZE, 
                        help=f'윈도우 크기(초) (기본: {WINDOW_SIZE})')
    parser.add_argument('--window-overlap', type=float, default=WINDOW_OVERLAP,
                        help=f'윈도우 오버랩(초) (기본: {WINDOW_OVERLAP})')
    parser.add_argument('--interval', type=float, default=5.0,
                        help='윈도우 발행 간격(초) (기본: 5.0)')
    
    args = parser.parse_args()
    
    # CSV 파일 존재 확인
    if not os.path.exists(args.csv):
        print(f"❌ CSV 파일을 찾을 수 없습니다: {args.csv}")
        sys.exit(1)
    
    if args.ir_csv and not os.path.exists(args.ir_csv):
        print(f"❌ IR CSV 파일을 찾을 수 없습니다: {args.ir_csv}")
        sys.exit(1)
    
    replay_csv(
        csv_path=args.csv,
        ir_csv_path=args.ir_csv,
        broker=args.broker,
        port=args.port,
        speed=args.speed,
        window_size=args.window_size,
        window_overlap=args.window_overlap,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
