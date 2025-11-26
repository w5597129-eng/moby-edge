"""Standalone inference worker that consumes sensor windows via MQTT."""
from __future__ import annotations

import copy
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import joblib
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

# [수정 1] 외부 feature_extractor 임포트 시도 구문을 삭제했습니다.
# 이제 무조건 아래에 정의된 extract_features 함수를 사용합니다.

import paho.mqtt.client as mqtt
try:
    import onnxruntime as ort
except Exception:
    ort = None

from inference_interface import (
    InferenceResultMessage,
    WindowMessage,
    SENSOR_FIELDS,
    USE_FREQUENCY_DOMAIN,
    current_timestamp_ns,
    model_result_topic,
    result_topic,
    WINDOW_TOPIC_ROOT,
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


DEFAULT_MODEL_CONFIGS: List[ModelConfig] = [
    ModelConfig(
        name="isolation_forest",
        sensor_type="accel_gyro",
        model_path=_resolve_path(
            "models/isolation_forest.joblib",
            "models/isolation_forest.pkl",
        ),
        scaler_path=_resolve_path(
            "models/scaler_if.joblib",
            "models/scaler_if.pkl",
        ),
        score_field="iforest_score",
        feature_pipeline="unit_norm",
        max_retries=3,
    ),
    ModelConfig(
        name="mlp_classifier",
        sensor_type="accel_gyro",
        model_path=_resolve_path(
            "models/mlp_classifier.onnx",
        ),
        scaler_path=_resolve_path(
            "models/scaler_mlp.pkl",
        ),
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


# PyTorch support removed: project uses ONNX / joblib/pickle models only.


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


# [수정 2] if extract_features is None: 조건문을 제거하고 함수를 직접 정의합니다.
# 이렇게 하면 외부 파일 유무와 상관없이 항상 이 '안전한' 버전이 사용됩니다.
def _legacy_extract_features(signal: Iterable[float], sampling_rate: float, use_freq_domain: bool = USE_FREQUENCY_DOMAIN) -> list:
    # 리스트를 numpy 배열로 확실하게 변환 (가장 중요한 부분)
    signal = np.asarray(list(signal), dtype=float)
    
    if signal.size < 2:
        feature_count = 11 if use_freq_domain else 5
        return [0.0] * feature_count

    abs_signal = np.abs(signal)
    max_val = float(np.max(abs_signal))
    abs_mean = float(np.mean(abs_signal))
    std = float(np.std(signal))
    peak_to_peak = float(np.ptp(signal))
    
    # 여기서 signal이 numpy 배열이므로 ** 2 연산이 정상 작동합니다.
    rms = float(np.sqrt(np.mean(signal ** 2)))
    
    crest_factor = max_val / rms if rms > 0 else 0.0
    impulse_factor = max_val / abs_mean if abs_mean > 0 else 0.0
    mean_val = float(np.mean(signal))
    time_features = [std, peak_to_peak, crest_factor, impulse_factor, mean_val]

    if not use_freq_domain:
        return time_features

    signal_centered = signal - np.mean(signal)
    spectrum = np.abs(rfft(signal_centered))
    freqs = rfftfreq(signal.size, 1.0 / sampling_rate) if signal.size > 0 else np.array([0.0])
    dominant_freq = float(freqs[np.argmax(spectrum)]) if spectrum.size > 0 else 0.0
    spectral_sum = float(np.sum(spectrum))
    spectral_centroid = float(np.sum(freqs * spectrum) / spectral_sum) if spectral_sum > 0 else 0.0
    spectral_energy = float(np.sum(spectrum ** 2))
    
    spectral_kurt = float(kurtosis(spectrum, fisher=False)) if spectrum.size > 1 else 3.0
    spectral_skewness = float(skew(spectrum)) if spectrum.size > 1 else 0.0
    spectral_kurt = 3.0 if np.isnan(spectral_kurt) else spectral_kurt
    spectral_skewness = 0.0 if np.isnan(spectral_skewness) else spectral_skewness
    spectral_std = float(np.std(spectrum))
    freq_features = [
        dominant_freq,
        spectral_centroid,
        spectral_energy,
        spectral_kurt,
        spectral_skewness,
        spectral_std,
    ]
    return time_features + freq_features


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
        model_path = self.config.model_path
        if model_path and os.path.exists(model_path) and os.path.splitext(model_path)[1].lower() == ".onnx":
            try:
                self.model = ONNXMLPWrapper(model_path)
            except Exception as exc:
                print(f"Failed to load ONNX model ({model_path}): {exc}")
                self.model = None
            loaded_model = None
        else:
            loaded_model = _load_artifact(self.config.model_path)
        if loaded_model is not None:
            # Loaded artifact is used directly. PyTorch checkpoint handling removed.
            self.model = loaded_model
        loaded_scaler = _load_artifact(self.config.scaler_path)
        if loaded_scaler is not None:
            self.scaler = loaded_scaler

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

    def _predict_proba_values(self, prepared: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        predict_fn = None
        for attr in ("predict_proba", "predict_proba_raw"):
            if hasattr(self.model, attr):
                predict_fn = getattr(self.model, attr)
                break
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
        probas = self._predict_proba_values(prepared)
        score = None
        if self.model is not None:
            try:
                score = float(self.model.score_samples(prepared)[0])
            except Exception as exc:
                print(f"Model score error ({self.config.name}):", exc)
        context_payload: Dict[str, Any] = copy.deepcopy(window_msg.context_payload) if isinstance(window_msg.context_payload, dict) else {}
        context_fields = context_payload.setdefault("fields", {})
        self._attach_probability_fields(context_fields, probas)
        if self.config.name == "mlp_classifier" and probas is not None and probas.size > 0:
            row = probas[0]
            red_probs = [row[i] for i in (2, 5) if i < len(row)]
            if red_probs:
                score = float(max(red_probs))
                context_fields[self.config.score_field] = score
        elif score is not None:
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
        context_payload = copy.deepcopy(window_msg.context_payload) if isinstance(window_msg.context_payload, dict) else {}
        context_fields = context_payload.setdefault("fields", {})
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

    def _build_feature_vector(self, window_msg: WindowMessage) -> np.ndarray:
        # Prefer the new multi-axis feature extractor (v17) when available.
        try:
            from feature_extractor import extract_features as multi_sensor_extract_features, FEATURE_CONFIG_V17
        except Exception:
            multi_sensor_extract_features = None
            FEATURE_CONFIG_V17 = None

        # If the multivariate extractor is available, build sensor-level dict (accel/gyro/pressure)
        if multi_sensor_extract_features is not None:
            wf = window_msg.window_fields or {}
            data_dict = {}
            sr = float(window_msg.sampling_rate_hz)

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

            # Pressure (optional)
            if 'fields_pressure_hpa' in wf and wf.get('fields_pressure_hpa') is not None:
                try:
                    arr = np.asarray(wf['fields_pressure_hpa'], dtype=float)
                    arr = arr[~np.isnan(arr)]
                    if arr.size > 0:
                        data_dict['pressure'] = arr
                except Exception:
                    pass

            # Temperature (if provided)
            if 'fields_temperature_c' in wf and wf.get('fields_temperature_c') is not None:
                try:
                    arr = np.asarray(wf['fields_temperature_c'], dtype=float)
                    arr = arr[~np.isnan(arr)]
                    if arr.size > 0:
                        data_dict['temperature'] = arr
                except Exception:
                    pass

            if data_dict:
                try:
                    feats_dict = multi_sensor_extract_features(data_dict, sr)
                    # Build deterministic feature order using FEATURE_CONFIG_V17 if available
                    ordered_keys = []
                    if FEATURE_CONFIG_V17 is not None:
                        for sensor_key in ('accel', 'gyro', 'pressure', 'temperature'):
                            for fname in FEATURE_CONFIG_V17.get(sensor_key, []):
                                ordered_keys.append(f"{sensor_key}_{fname}")
                    else:
                        ordered_keys = sorted(feats_dict.keys())

                    vec = [float(feats_dict.get(k, 0.0)) for k in ordered_keys]
                    return np.asarray(vec, dtype=float).reshape(1, -1)
                except Exception:
                    # Fall back to legacy per-field extraction below
                    pass

        # Legacy per-field extractor fallback (unchanged behavior)
        feats = []
        for field in SENSOR_FIELDS:
            sig = window_msg.window_fields.get(field)
            # [확인] 리스트가 들어와도 numpy 배열로 변환합니다.
            arr = np.asarray(sig, dtype=float) if sig is not None else np.asarray([])
            if arr.size < 2:
                feat_len = 11 if USE_FREQUENCY_DOMAIN else 5
                feats.extend([0.0] * feat_len)
            else:
                feats.extend(_legacy_extract_features(arr, window_msg.sampling_rate_hz, USE_FREQUENCY_DOMAIN))
        return np.asarray(feats, dtype=float).reshape(1, -1)

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
    def __init__(self, broker: str = "localhost", port: int = 1883):
        self.broker = broker
        self.port = port
        self.engine = build_default_engine()
        self.client = _make_mqtt_client("sensor_inference_worker")
        self.client.on_message = self._on_message
        self.client.on_connect = self._on_connect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT broker at {self.broker}:{self.port}")
            client.subscribe(f"{WINDOW_TOPIC_ROOT}/#")
        else:
            print(f"Failed to connect to MQTT broker, return code {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            window_msg = WindowMessage.from_payload(payload)
            results = self.engine.process_window(window_msg)
            for result in results:
                topic = result_topic(result.sensor_type)
                if result.model_name:
                    topic = model_result_topic(result.sensor_type, result.model_name)
                client.publish(topic, json.dumps(result.to_payload()))
                print(
                    f"INFERENCE | {result.sensor_type} | model={result.model_name} "
                    f"score={result.score} label={result.label}"
                )
        except Exception as exc:
            print(f"Inference MQTT handler error: {exc}")

    def start(self):
        print(f"Starting Inference Worker on {self.broker}:{self.port}...")
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_forever()


def main():
    worker = MQTTInferenceWorker()
    worker.start()


if __name__ == "__main__":
    main()