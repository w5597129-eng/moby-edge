"""Standalone inference worker that consumes sensor windows via MQTT."""
from __future__ import annotations

import json
import os
import pickle
import signal
import time
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import joblib
import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    pass

import paho.mqtt.client as mqtt

from inference_interface import (
    InferenceResultMessage,
    WindowMessage,
    SENSOR_FIELDS,
    USE_FREQUENCY_DOMAIN,
    current_timestamp_ns,
    model_result_topic,
    result_topic,
    WINDOW_TOPIC_ROOT,
    FEATURE_ORDER_V19, # V19 Import
    EXPECTED_FEATURE_COUNT,
)


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
    result_topic_override: Optional[str] = None
    score_field: str = "anomaly_score"
    feature_pipeline: str = "identity"
    max_retries: int = 2
    probability_field_names: Optional[List[str]] = None

    def result_topic(self) -> str:
        if self.result_topic_override:
            return self.result_topic_override
        return model_result_topic(self.sensor_type, self.name)


FEATURE_DEBUG = os.getenv("FEATURE_DEBUG", "0").lower() in ("1", "true", "yes")
FEATURE_DEBUG_MAX_ITEMS = int(os.getenv("FEATURE_DEBUG_MAX_ITEMS", "8"))

def _formatted_vector(arr: np.ndarray, limit: int = FEATURE_DEBUG_MAX_ITEMS) -> str:
    flat = np.asarray(arr).flatten()
    values = ", ".join(f"{v:.4f}" for v in flat[:limit])
    suffix = "..." if flat.size > limit else ""
    return f"{values}{suffix}"

def _debug_feature_vector(stage: str, config_name: str, vector: np.ndarray) -> None:
    if not FEATURE_DEBUG:
        return
    print(
        f"[FEAT] {stage} | {config_name} | len={np.asarray(vector).size}"
        f" | data={_formatted_vector(vector)}"
    )

DEFAULT_MODEL_CONFIGS: List[ModelConfig] = [
    ModelConfig(
        name="isolation_forest",
        sensor_type="accel_gyro",
        model_path="models/isolation_forest.joblib",
        scaler_path="models/scaler_if.joblib",
        score_field="iforest_score",
        feature_pipeline="identity",
        max_retries=3,
    ),
    ModelConfig(
        name="lgbm_classifier",
        sensor_type="accel_gyro",
        model_path=_resolve_path(
            "models/lgbm_classifier.joblib",
        ),
        scaler_path=None, 
        score_field="lgbm_score",
        feature_pipeline="identity",
        max_retries=2,
        probability_field_names=[
            "lgbm_prob_normal",
            "lgbm_prob_s1_yellow",
            "lgbm_prob_s1_red",
            "lgbm_prob_s2_yellow",
            "lgbm_prob_s2_red",
        ],
    ),
]


def _make_mqtt_client(client_id: str) -> mqtt.Client:
    try:
        return mqtt.Client(client_id=client_id)
    except Exception as e:
        try:
            return mqtt.Client(client_id=client_id, callback_api_version=1)
        except Exception:
            try:
                return mqtt.Client(client_id=client_id, userdata=None, protocol=mqtt.MQTTv311)
            except Exception:
                raise e


FEATURE_PIPELINES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

def feature_pipeline(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name not in FEATURE_PIPELINES:
        raise KeyError(f"Unknown feature pipeline: {name}")
    return FEATURE_PIPELINES[name]

def register_feature_pipeline(name: str):
    def decorator(func: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        FEATURE_PIPELINES[name] = func
        return func
    return decorator

@register_feature_pipeline("identity")
def _identity_pipeline(features: np.ndarray) -> np.ndarray:
    return features

@register_feature_pipeline("unit_norm")
def _unit_norm_pipeline(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features)
    if norm == 0:
        return features
    return features / norm

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

class ModelRunner:
    def __init__(self, config: ModelConfig, model: Optional[Any] = None, scaler: Optional[Any] = None):
        self.config = config
        self.model = model
        self.scaler = scaler
        if self.model is None and self.config.model_path:
            self.reload_artifacts()
        elif self.scaler is None and self.config.scaler_path:
            self.scaler = _load_artifact(self.config.scaler_path)

    def reload_artifacts(self):
        loaded_model = _load_artifact(self.config.model_path)
        if loaded_model is not None:
            self.model = loaded_model
            
        loaded_scaler = _load_artifact(self.config.scaler_path)
        if loaded_scaler is not None:
            self.scaler = loaded_scaler

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        prepared = np.asarray(features, dtype=float)
        if prepared.ndim == 1:
            prepared = prepared.reshape(1, -1)
        _debug_feature_vector("raw", self.config.name, prepared)
        pipeline_fn = feature_pipeline(self.config.feature_pipeline)
        prepared = pipeline_fn(prepared)
        _debug_feature_vector("after_pipeline", self.config.name, prepared)
        if self.scaler is not None:
            _debug_feature_vector("before_scaling", self.config.name, prepared)
            try:
                prepared = self.scaler.transform(prepared)
            except Exception as exc:
                print(f"Scaler transform error ({self.config.name}):", exc)
            _debug_feature_vector("after_scaling", self.config.name, prepared)
        return prepared

    def _predict_proba_values(self, prepared: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        
        predict_fn = None
        if hasattr(self.model, "predict_proba"):
             predict_fn = self.model.predict_proba
        
        if predict_fn is None:
            return None
            
        try:
            res = predict_fn(prepared)
            arr = np.asarray(res, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception as exc:
            print(f"Probability extraction error ({self.config.name}):", exc)
            return None

    def _attach_probability_fields(self, fields: Dict[str, Any], probas: Optional[np.ndarray]) -> None:
        if probas is None or probas.size == 0:
            return
        names = self.config.probability_field_names or []
        row = probas[0]
        for idx, value in enumerate(row):
            key = names[idx] if idx < len(names) else f"{self.config.name}_prob_{idx}"
            fields[key] = float(value)

    def _run_once(self, window_msg: WindowMessage, features: np.ndarray) -> InferenceResultMessage:
        prepared = self._prepare_features(features)
        
        score = None
        probas = None
        
        if self.config.name == "isolation_forest":
             if self.model is not None:
                try:
                    score = float(self.model.decision_function(prepared)[0])
                except Exception as exc:
                    print(f"Model score error ({self.config.name}):", exc)
        else:
            probas = self._predict_proba_values(prepared)
            pass

            
        context_payload: Dict[str, Any] = {"fields": {}}
        context_fields = context_payload["fields"]
        self._attach_probability_fields(context_fields, probas)
        
        if score is not None:
            context_fields[self.config.score_field] = score
            
        context_payload.setdefault("timestamp_ns", window_msg.timestamp_ns or current_timestamp_ns())
        
        return InferenceResultMessage(
            sensor_type=window_msg.sensor_type,
            score=score,
            model_name=self.config.name,
            timestamp_ns=current_timestamp_ns(),
            context_payload=context_payload,
        )

    def run(self, window_msg: WindowMessage, features: np.ndarray) -> InferenceResultMessage:
        attempts = 0
        last_error: Optional[Exception] = None
        while attempts < max(1, self.config.max_retries):
            try:
                return self._run_once(window_msg, features)
            except Exception as exc:
                last_error = exc
                attempts += 1
                print(f"Model {self.config.name} inference failed (attempt {attempts}): {exc}")
                self.reload_artifacts()
        context_payload = {"fields": {}}
        context_fields = context_payload["fields"]
        context_fields[f"{self.config.name}_error"] = str(last_error) if last_error else "unknown"
        return InferenceResultMessage(
            sensor_type=window_msg.sensor_type,
            score=None,
            model_name=self.config.name,
            timestamp_ns=current_timestamp_ns(),
            context_payload=context_payload,
        )

    def result_topic(self) -> str:
        return self.config.result_topic()


class InferenceEngine:
    def __init__(self, runners: Optional[List[ModelRunner]] = None):
        self.runners = runners or []
        self.v19_available = False # Renamed logic for clarity
        self.feature_extraction_mode = "legacy"
        self._check_v19_availability()

    def _check_v19_availability(self):
        """Check if V19 feature extractor is available (required)."""
        try:
            # Try src first, then root directory
            try:
                from feature_extractor import extract_features, FEATURE_CONFIG_V19
            except ImportError:
                import sys
                import os
                src_dir = os.path.dirname(os.path.abspath(__file__))
                root_dir = os.path.dirname(src_dir)
                if root_dir not in sys.path:
                    sys.path.insert(0, root_dir)
                from feature_extractor import extract_features, FEATURE_CONFIG_V19
            
            self.v19_available = True
            self.feature_extraction_mode = "v19"
            print("[INFERENCE_ENGINE] ✅ V19 feature extractor loaded (REQUIRED)")
        except Exception as e:
            self.v19_available = False
            self.feature_extraction_mode = "unavailable"
            print(f"[INFERENCE_ENGINE] ❌ FATAL: V19 feature extractor not available: {e}")
            print("[INFERENCE_ENGINE] ❌ V19 is now REQUIRED")
            raise RuntimeError(f"V19 feature extractor is mandatory but failed to load: {e}") from e

    def _build_feature_vector(self, window_msg: WindowMessage) -> np.ndarray:
        """Build feature vector using V19 feature extractor."""
        if not self.v19_available:
            raise RuntimeError("[FEATURE_VECTOR] ❌ V19 feature extractor not available")
        
        try:
            # Try src first, then root directory
            try:
                from feature_extractor import extract_features as multi_sensor_extract_features, FEATURE_CONFIG_V19
            except ImportError:
                import sys
                import os
                src_dir = os.path.dirname(os.path.abspath(__file__))
                root_dir = os.path.dirname(src_dir)
                if root_dir not in sys.path:
                    sys.path.insert(0, root_dir)
                from feature_extractor import extract_features as multi_sensor_extract_features, FEATURE_CONFIG_V19
        except Exception as e:
            raise RuntimeError(f"[FEATURE_VECTOR] ❌ Failed to import V19: {e}") from e

        wf = window_msg.window_fields or {}
        data_dict = {}
        
        timestamps = wf.get('timestamp_ns')
        if timestamps and len(timestamps) > 1:
            try:
                ts_array = np.asarray(timestamps, dtype=np.int64)
                duration_sec = (ts_array[-1] - ts_array[0]) / 1e9
                if duration_sec > 0:
                    sr = (len(ts_array) - 1) / duration_sec
                else:
                    sr = 12.8
            except Exception:
                sr = 12.8
        else:
            sr = 12.8
        
        print(f"[DEBUG] Actual SR from timestamps: {sr:.2f} Hz (samples={len(timestamps) if timestamps else 0})")

        # Accel 3-axis
        accel_cols = ['fields_accel_x', 'fields_accel_y', 'fields_accel_z']
        if all(col in wf and wf[col] is not None for col in accel_cols):
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
        if all(col in wf and wf[col] is not None for col in gyro_cols):
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

        # IR Counter data (avg_cycle_ms, last_cycle_ms)
        avg_cycle_col = 'fields_avg_cycle_ms'
        last_cycle_col = 'fields_last_cycle_ms'
        if avg_cycle_col in wf and last_cycle_col in wf:
            try:
                avg_arr = np.asarray(wf[avg_cycle_col], dtype=float)
                last_arr = np.asarray(wf[last_cycle_col], dtype=float)
                # Filter out NaN values
                avg_valid = avg_arr[~np.isnan(avg_arr)]
                last_valid = last_arr[~np.isnan(last_arr)]
                min_len = min(len(avg_valid), len(last_valid))
                if min_len > 0:
                    data_dict['ir_avg'] = avg_valid[:min_len]
                    data_dict['ir_last'] = last_valid[:min_len]
            except Exception:
                pass

        if data_dict:
            try:
                feats_dict = multi_sensor_extract_features(data_dict, sr)
                # [수정] V19 순서 사용
                vec = [float(feats_dict.get(k, 0.0)) for k in FEATURE_ORDER_V19]
                result = np.asarray(vec, dtype=float).reshape(1, -1)
                
                if result.shape[1] != EXPECTED_FEATURE_COUNT:
                    print(f"[FEATURE_VECTOR] ⚠️  V19 feature count mismatch: got {result.shape[1]}, expected {EXPECTED_FEATURE_COUNT}")
                
                _debug_feature_vector("v19_output", window_msg.sensor_type, result)
                return result
            except Exception as e:
                print(f"[FEATURE_VECTOR] ❌ V19 extraction failed: {e}")
                raise RuntimeError(f"Feature extraction failed with V19: {e}") from e

        raise RuntimeError("[FEATURE_VECTOR] ❌ No valid sensor data in window")

    def process_window(self, window_msg: WindowMessage) -> List[InferenceResultMessage]:
        features = self._build_feature_vector(window_msg)
        results: List[InferenceResultMessage] = []
        for runner in self.runners:
            results.append(runner.run(window_msg, features))
        return results


def build_default_engine() -> InferenceEngine:
    runners = [ModelRunner(config) for config in DEFAULT_MODEL_CONFIGS]
    return InferenceEngine(runners)


class MQTTInferenceWorker:
    def __init__(self, broker: str = None, port: int = None):
        from pathlib import Path
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        except ImportError:
            pass
        broker = broker or os.getenv("MQTT_BROKER", "192.168.80.234")
        port = port or int(os.getenv("MQTT_PORT", "1883"))
        self.broker = broker
        self.port = port
        self.engine = build_default_engine()
        self.client = _make_mqtt_client("sensor_inference_worker")
        self.client.on_message = self._on_message
        self.client.on_connect = self._on_connect
        
        self.message_queue = queue.Queue(maxsize=100) 
        self.is_running = False
        self.worker_thread = None
        
        self._log_initialization()

    def _on_connect(self, client, userdata, flags, rc, *args):
        if isinstance(rc, int):
            rc_value = rc
        else:
            rc_value = rc.value if hasattr(rc, 'value') else 0
        if rc_value == 0:
            print(f"Connected to MQTT broker at {self.broker}:{self.port}")
            client.subscribe(f"{WINDOW_TOPIC_ROOT}/#")
            print(f"Subscribed to {WINDOW_TOPIC_ROOT}/#")
        else:
            print(f"Failed to connect to MQTT broker, return code {rc_value}")

    def _log_initialization(self):
        print("\n" + "="*70)
        print("INFERENCE WORKER INITIALIZATION")
        print("="*70)
        print(f"Feature Extraction Mode: {self.engine.feature_extraction_mode.upper()}")
        print(f"Models Loaded: {len(self.engine.runners)}")
        for runner in self.engine.runners:
            print(f"  - {runner.config.name} ({runner.config.sensor_type})")
        print("="*70 + "\n")

    def _on_message(self, client, userdata, msg):
        try:
            self.message_queue.put_nowait(msg)
        except queue.Full:
            print(f"[WARNING] Message queue full! Dropping message from {msg.topic}")
        except Exception as exc:
            print(f"Enqueuing error: {exc}")

    def _processing_loop(self):
        print("[WORKER] Processing thread started.")
        while self.is_running:
            try:
                msg = self.message_queue.get(timeout=1.0) 
            except queue.Empty:
                continue

            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                window_msg = WindowMessage.from_payload(payload)
                results = self.engine.process_window(window_msg)
                for result in results:
                    topic = result_topic(result.sensor_type)
                    if result.model_name:
                        topic = model_result_topic(result.sensor_type, result.model_name)
                    self.client.publish(topic, json.dumps(result.to_payload()))
                self._print_monitoring(results)
                
            except Exception as exc:
                print(f"Processing error in worker thread: {exc}")
            finally:
                self.message_queue.task_done()
        print("[WORKER] Processing thread stopped.")

    def _print_monitoring(self, results: List[InferenceResultMessage]):
        iforest_score = None
        lgbm_prob = {}
        lgbm_label = "N/A"
        
        for result in results:
            fields = (result.context_payload or {}).get("fields", {})
            
            if result.model_name == "isolation_forest":
                iforest_score = fields.get("iforest_score", result.score)
            
            elif result.model_name == "lgbm_classifier":
                # 5-class Probabilities
                lgbm_prob = {
                    "NORMAL": fields.get("lgbm_prob_normal", 0.0),
                    "S1_YELLOW": fields.get("lgbm_prob_s1_yellow", 0.0),
                    "S1_RED": fields.get("lgbm_prob_s1_red", 0.0),
                    "S2_YELLOW": fields.get("lgbm_prob_s2_yellow", 0.0),
                    "S2_RED": fields.get("lgbm_prob_s2_red", 0.0),
                }
                lgbm_label = max(lgbm_prob, key=lgbm_prob.get)

        print("\n" + "=" * 70)
        print(f"{'INFERENCE MONITORING':^70}")
        print("=" * 70)
        
        # Isolation Forest
        if iforest_score is not None:
            anomaly_status = "🔴 ANOMALY" if iforest_score < 0 else "🟢 NORMAL"
            print(f"[Isolation Forest] Score: {iforest_score:+.4f}  |  {anomaly_status}")
        else:
            print(f"[Isolation Forest] Score: N/A")
        
        print("-" * 70)
        
        # LightGBM 결과 출력
        # Label emoji mapping
        emoji_map = {
            "NORMAL": "🟢",
            "S1_YELLOW": "🟡",
            "S1_RED": "🔴",
            "S2_YELLOW": "🟡",
            "S2_RED": "🔴",
            "N/A": "⚪"
        }
        emoji = emoji_map.get(lgbm_label, "⚪")
        
        print(f"[LightGBM Diagnosis] {emoji} {lgbm_label}")
        print(f"    Normal    : {lgbm_prob.get('NORMAL', 0)*100:6.2f}%")
        print(f"    S1 Warning: {lgbm_prob.get('S1_YELLOW', 0)*100:6.2f}%  |  S1 Fault: {lgbm_prob.get('S1_RED', 0)*100:6.2f}%")
        print(f"    S2 Warning: {lgbm_prob.get('S2_YELLOW', 0)*100:6.2f}%  |  S2 Fault: {lgbm_prob.get('S2_RED', 0)*100:6.2f}%")
        
        print("=" * 70 + "\n")

    def start(self):
        print(f"Starting Inference Worker on {self.broker}:{self.port}...")
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._processing_loop, name="InferenceWorker")
        self.worker_thread.start()
        
        print(f"Attempting to connect to MQTT broker...")
        try:
            self.client.connect(self.broker, self.port, 60)
            print(f"Connection initiated, starting loop...")
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.is_running = False
            return

        self.client.loop_start()
        print(f"MQTT loop started, waiting for messages...")
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
        self.is_running = False
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
        if self.worker_thread and self.worker_thread.is_alive():
            print("[INFERENCE_WORKER] Waiting for processing thread to finish...")
            self.worker_thread.join(timeout=5.0)
        print("\n[INFERENCE_WORKER] Clean exit.")


stop_flag = False

def handle_stop(sig, frame):
    global stop_flag
    stop_flag = True

signal.signal(signal.SIGINT, handle_stop)
signal.signal(signal.SIGTERM, handle_stop)


def main():
    worker = MQTTInferenceWorker()
    try:
        worker.start()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()


if __name__ == "__main__":
    main()