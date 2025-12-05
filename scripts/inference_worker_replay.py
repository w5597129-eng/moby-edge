#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay 전용 Inference Worker

replay_csv_data.py가 발행하는 윈도우 메시지만 구독하여 추론합니다.
별도의 토픽(factory/inference/replay/windows)을 사용하여 
실제 센서 데이터와 분리됩니다.

사용법:
    1. 이 워커 실행:
       python scripts/inference_worker_replay.py
    
    2. CSV 재생 (--replay 옵션 사용):
       python scripts/replay_csv_data.py --csv data/1205_accel_gyro_s1_red.csv --replay
"""

from __future__ import annotations

import json
import os
import pickle
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import joblib
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

import paho.mqtt.client as mqtt
try:
    import onnxruntime as ort
except Exception:
    ort = None

from src.inference_interface import (
    InferenceResultMessage,
    WindowMessage,
    SENSOR_FIELDS,
    USE_FREQUENCY_DOMAIN,
    current_timestamp_ns,
    FEATURE_ORDER_V18,
    EXPECTED_FEATURE_COUNT,
    RESULT_TOPIC_ROOT,
    model_result_topic,
    result_topic,
)

# Replay 전용 토픽 (구독만)
REPLAY_WINDOW_TOPIC_ROOT = "factory/inference/replay/windows"
# 결과는 기존 토픽으로 발행 (RESULT_TOPIC_ROOT 사용)


def _resolve_path(*candidates: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


@dataclass
class ModelConfig:
    name: str
    sensor_type: str
    model_path: str
    scaler_path: Optional[str] = None
    score_field: str = "anomaly_score"
    feature_pipeline: str = "identity"
    max_retries: int = 2
    probability_field_names: Optional[List[str]] = None


FEATURE_PIPELINES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

def register_feature_pipeline(name: str):
    def decorator(func):
        FEATURE_PIPELINES[name] = func
        return func
    return decorator

@register_feature_pipeline("identity")
def _identity_pipeline(features: np.ndarray) -> np.ndarray:
    return features

def feature_pipeline(name: str):
    return FEATURE_PIPELINES.get(name, _identity_pipeline)


def _load_artifact(path: Optional[str]):
    if not path:
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception as exc:
            print(f"Artifact load error ({path}):", exc)
            return None


class ONNXMLPWrapper:
    def __init__(self, onnx_path: str, providers: Optional[list] = None):
        if ort is None:
            raise RuntimeError("onnxruntime not available")
        self.onnx_path = onnx_path
        chosen_providers = providers if providers is not None else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=chosen_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X is None:
            return np.zeros((0, 2))
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        res = self.session.run([self.output_name], {self.input_name: arr})[0]
        return np.asarray(res)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        proba = self._predict_proba(X)
        if proba.size == 0:
            return np.array([])
        return np.linalg.norm(proba, axis=1)

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        return self._predict_proba(X)


# 모델 설정
DEFAULT_MODEL_CONFIGS: List[ModelConfig] = [
    ModelConfig(
        name="isolation_forest",
        sensor_type="accel_gyro",
        model_path=_resolve_path("models/isolation_forest.joblib"),
        scaler_path=_resolve_path("models/scaler_if.joblib"),
        score_field="iforest_score",
        feature_pipeline="identity",
        max_retries=3,
    ),
    ModelConfig(
        name="mlp_classifier",
        sensor_type="accel_gyro",
        model_path=_resolve_path("models/mlp_classifier.onnx"),
        scaler_path=_resolve_path("models/scaler_mlp.pkl"),
        score_field="mlp_score",
        feature_pipeline="identity",
        max_retries=2,
        probability_field_names=[
            "mlp_s1_prob_normal",
            "mlp_s1_prob_yellow",
            "mlp_s1_prob_red",
            "mlp_s2_prob_normal",
            "mlp_s2_prob_yellow",
            "mlp_s2_prob_red",
        ],
    ),
]


class ModelRunner:
    def __init__(self, config: ModelConfig, model=None, scaler=None):
        self.config = config
        self.model = model
        self.scaler = scaler
        if self.model is None and self.config.model_path:
            self.reload_artifacts()
        elif self.scaler is None and self.config.scaler_path:
            self.scaler = _load_artifact(self.config.scaler_path)

    def reload_artifacts(self):
        model_path = self.config.model_path
        if model_path and os.path.exists(model_path) and model_path.endswith(".onnx"):
            try:
                self.model = ONNXMLPWrapper(model_path)
            except Exception as exc:
                print(f"Failed to load ONNX model ({model_path}): {exc}")
                self.model = None
        else:
            self.model = _load_artifact(self.config.model_path)
        self.scaler = _load_artifact(self.config.scaler_path)

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        prepared = np.asarray(features, dtype=float)
        if prepared.ndim == 1:
            prepared = prepared.reshape(1, -1)
        pipeline_fn = feature_pipeline(self.config.feature_pipeline)
        prepared = pipeline_fn(prepared)
        if self.scaler is not None:
            try:
                prepared = self.scaler.transform(prepared)
            except Exception as exc:
                print(f"Scaler transform error ({self.config.name}):", exc)
        return prepared

    def _predict_proba_values(self, prepared: np.ndarray):
        if self.model is None:
            return None
        for attr in ("predict_proba", "predict_proba_raw"):
            if hasattr(self.model, attr):
                try:
                    res = getattr(self.model, attr)(prepared)
                    arr = np.asarray(res, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    return arr
                except Exception:
                    pass
        return None

    def _attach_probability_fields(self, fields: Dict[str, Any], probas):
        if probas is None or probas.size == 0:
            return
        names = self.config.probability_field_names or []
        row = probas[0]
        for idx, value in enumerate(row):
            key = names[idx] if idx < len(names) else f"{self.config.name}_prob_{idx}"
            fields[key] = float(value)

    def run(self, window_msg: WindowMessage, features: np.ndarray) -> InferenceResultMessage:
        prepared = self._prepare_features(features)
        probas = self._predict_proba_values(prepared)
        score = None
        
        if self.model is not None:
            try:
                score = float(self.model.score_samples(prepared)[0])
            except Exception as exc:
                print(f"Model score error ({self.config.name}):", exc)
        
        context_payload: Dict[str, Any] = {"fields": {}}
        context_fields = context_payload["fields"]
        self._attach_probability_fields(context_fields, probas)
        
        if self.config.name == "mlp_classifier" and probas is not None and probas.size > 0:
            row = probas[0]
            red_probs = [row[i] for i in (2, 5) if i < len(row)]
            if red_probs:
                score = float(max(red_probs))
                context_fields[self.config.score_field] = score
        elif score is not None:
            context_fields[self.config.score_field] = score
        
        return InferenceResultMessage(
            sensor_type=window_msg.sensor_type,
            score=score,
            model_name=self.config.name,
            timestamp_ns=current_timestamp_ns(),
            context_payload=context_payload,
        )


class InferenceEngine:
    def __init__(self, runners: Optional[List[ModelRunner]] = None):
        self.runners = runners or []
        self._load_feature_extractor()

    def _load_feature_extractor(self):
        try:
            from feature_extractor import extract_features, FEATURE_CONFIG_V18
            self.extract_features = extract_features
            print("[REPLAY_WORKER] ✅ V18 feature extractor loaded")
        except ImportError:
            try:
                from src.feature_extractor import extract_features, FEATURE_CONFIG_V18
                self.extract_features = extract_features
                print("[REPLAY_WORKER] ✅ V18 feature extractor loaded (from src)")
            except Exception as e:
                print(f"[REPLAY_WORKER] ❌ Feature extractor not found: {e}")
                self.extract_features = None

    def _build_feature_vector(self, window_msg: WindowMessage) -> np.ndarray:
        if self.extract_features is None:
            raise RuntimeError("Feature extractor not available")
        
        wf = window_msg.window_fields or {}
        data_dict = {}
        
        # 샘플링 레이트 계산
        timestamps = wf.get('timestamp_ns')
        if timestamps and len(timestamps) > 1:
            try:
                ts_array = np.asarray(timestamps, dtype=np.int64)
                duration_sec = (ts_array[-1] - ts_array[0]) / 1e9
                sr = (len(ts_array) - 1) / duration_sec if duration_sec > 0 else 12.8
            except Exception:
                sr = 12.8
        else:
            sr = 12.8

        # Accel 3-axis
        accel_cols = ['fields_accel_x', 'fields_accel_y', 'fields_accel_z']
        if all(col in wf and wf[col] for col in accel_cols):
            try:
                arrays = [np.asarray(wf[col], dtype=float) for col in accel_cols]
                min_len = min(a.size for a in arrays)
                if min_len >= 2:
                    stacked = np.column_stack([a[-min_len:] for a in arrays])
                    valid_mask = ~np.isnan(stacked).any(axis=1)
                    if valid_mask.sum() > 0:
                        data_dict['accel'] = stacked[valid_mask]
            except Exception:
                pass

        # Gyro 3-axis
        gyro_cols = ['fields_gyro_x', 'fields_gyro_y', 'fields_gyro_z']
        if all(col in wf and wf[col] for col in gyro_cols):
            try:
                arrays = [np.asarray(wf[col], dtype=float) for col in gyro_cols]
                min_len = min(a.size for a in arrays)
                if min_len >= 2:
                    stacked = np.column_stack([a[-min_len:] for a in arrays])
                    valid_mask = ~np.isnan(stacked).any(axis=1)
                    if valid_mask.sum() > 0:
                        data_dict['gyro'] = stacked[valid_mask]
            except Exception:
                pass

        # IR Counter
        if 'fields_avg_cycle_ms' in wf and 'fields_last_cycle_ms' in wf:
            try:
                avg_arr = np.asarray(wf['fields_avg_cycle_ms'], dtype=float)
                last_arr = np.asarray(wf['fields_last_cycle_ms'], dtype=float)
                avg_valid = avg_arr[~np.isnan(avg_arr)]
                last_valid = last_arr[~np.isnan(last_arr)]
                min_len = min(len(avg_valid), len(last_valid))
                if min_len > 0:
                    data_dict['ir_avg'] = avg_valid[:min_len]
                    data_dict['ir_last'] = last_valid[:min_len]
            except Exception:
                pass

        if data_dict:
            feats_dict = self.extract_features(data_dict, sr)
            vec = [float(feats_dict.get(k, 0.0)) for k in FEATURE_ORDER_V18]
            return np.asarray(vec, dtype=float).reshape(1, -1)
        
        raise RuntimeError("No valid sensor data in window")

    def process_window(self, window_msg: WindowMessage) -> List[InferenceResultMessage]:
        features = self._build_feature_vector(window_msg)
        return [runner.run(window_msg, features) for runner in self.runners]


def _make_mqtt_client(client_id: str) -> mqtt.Client:
    try:
        return mqtt.Client(client_id=client_id)
    except TypeError:
        try:
            return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
        except Exception:
            return mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)


class ReplayInferenceWorker:
    def __init__(self, broker: str = None, port: int = None):
        # .env 파일에서 환경변수 로드
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        except ImportError:
            pass
        
        self.broker = broker or os.getenv("MQTT_BROKER", "192.168.80.234")
        self.port = port or int(os.getenv("MQTT_PORT", "1883"))
        
        # 모델 로드
        runners = [ModelRunner(config) for config in DEFAULT_MODEL_CONFIGS]
        self.engine = InferenceEngine(runners)
        
        # MQTT 클라이언트
        self.client = _make_mqtt_client("replay_inference_worker")
        self.client.on_message = self._on_message
        self.client.on_connect = self._on_connect
        
        self._log_initialization()

    def _on_connect(self, client, userdata, flags, rc, *args):
        rc_value = rc if isinstance(rc, int) else (rc.value if hasattr(rc, 'value') else 0)
        if rc_value == 0:
            print(f"✅ Connected to MQTT broker at {self.broker}:{self.port}")
            client.subscribe(f"{REPLAY_WINDOW_TOPIC_ROOT}/#")
            print(f"📥 Subscribed to {REPLAY_WINDOW_TOPIC_ROOT}/#")
        else:
            print(f"❌ Failed to connect, return code {rc_value}")

    def _log_initialization(self):
        print("\n" + "=" * 70)
        print(f"{'REPLAY INFERENCE WORKER':^70}")
        print("=" * 70)
        print(f"📥 Subscribe Topic: {REPLAY_WINDOW_TOPIC_ROOT}/#")
        print(f"📤 Publish Topic:   {RESULT_TOPIC_ROOT}/... (기존과 동일)")
        print(f"Models Loaded: {len(self.engine.runners)}")
        for runner in self.engine.runners:
            status = "✅" if runner.model else "❌"
            print(f"  {status} {runner.config.name}")
        print("=" * 70 + "\n")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            window_msg = WindowMessage.from_payload(payload)
            results = self.engine.process_window(window_msg)
            
            for result in results:
                # 기존 inference_worker와 동일한 토픽으로 발행
                topic = result_topic(result.sensor_type)
                if result.model_name:
                    topic = model_result_topic(result.sensor_type, result.model_name)
                client.publish(topic, json.dumps(result.to_payload()))
            
            self._print_monitoring(results)
        except Exception as exc:
            print(f"❌ Inference error: {exc}")

    def _print_monitoring(self, results: List[InferenceResultMessage]):
        iforest_score = None
        s1_result = {"normal": 0.0, "yellow": 0.0, "red": 0.0, "label": "N/A"}
        s2_result = {"normal": 0.0, "yellow": 0.0, "red": 0.0, "label": "N/A"}
        
        for result in results:
            fields = (result.context_payload or {}).get("fields", {})
            
            if result.model_name == "isolation_forest":
                iforest_score = fields.get("iforest_score", result.score)
            
            elif result.model_name == "mlp_classifier":
                s1_result = {
                    "normal": fields.get("mlp_s1_prob_normal", 0.0),
                    "yellow": fields.get("mlp_s1_prob_yellow", 0.0),
                    "red": fields.get("mlp_s1_prob_red", 0.0),
                }
                s1_result["label"] = max(s1_result, key=lambda k: s1_result[k] if k != "label" else -1)
                
                s2_result = {
                    "normal": fields.get("mlp_s2_prob_normal", 0.0),
                    "yellow": fields.get("mlp_s2_prob_yellow", 0.0),
                    "red": fields.get("mlp_s2_prob_red", 0.0),
                }
                s2_result["label"] = max(s2_result, key=lambda k: s2_result[k] if k != "label" else -1)
        
        # 출력
        print("\n" + "=" * 70)
        print(f"{'REPLAY INFERENCE RESULT':^70}")
        print("=" * 70)
        
        if iforest_score is not None:
            status = "🔴 ANOMALY" if iforest_score < 0 else "🟢 NORMAL"
            print(f"[Isolation Forest] Score: {iforest_score:+.4f}  |  {status}")
        
        print("-" * 70)
        
        emoji = {"normal": "🟢", "yellow": "🟡", "red": "🔴"}.get(s1_result["label"], "⚪")
        print(f"[S1 Classification] {emoji} {s1_result['label'].upper()}")
        print(f"    Normal: {s1_result['normal']*100:6.2f}%  |  Yellow: {s1_result['yellow']*100:6.2f}%  |  Red: {s1_result['red']*100:6.2f}%")
        
        print("-" * 70)
        
        emoji = {"normal": "🟢", "yellow": "🟡", "red": "🔴"}.get(s2_result["label"], "⚪")
        print(f"[S2 Classification] {emoji} {s2_result['label'].upper()}")
        print(f"    Normal: {s2_result['normal']*100:6.2f}%  |  Yellow: {s2_result['yellow']*100:6.2f}%  |  Red: {s2_result['red']*100:6.2f}%")
        
        print("=" * 70 + "\n")

    def start(self):
        print(f"🚀 Starting Replay Inference Worker...")
        try:
            self.client.connect(self.broker, self.port, 60)
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return
        
        self.client.loop_start()
        
        global stop_flag
        try:
            while not stop_flag:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        global stop_flag
        stop_flag = True
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
        print("\n[REPLAY_WORKER] Clean exit.")


stop_flag = False

def handle_stop(sig, frame):
    global stop_flag
    stop_flag = True

signal.signal(signal.SIGINT, handle_stop)
signal.signal(signal.SIGTERM, handle_stop)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Replay 전용 Inference Worker")
    parser.add_argument('--broker', default=None, help='MQTT 브로커 주소')
    parser.add_argument('--port', type=int, default=None, help='MQTT 포트')
    args = parser.parse_args()
    
    worker = ReplayInferenceWorker(broker=args.broker, port=args.port)
    worker.start()


if __name__ == "__main__":
    main()
