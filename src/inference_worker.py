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
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None
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
    label_field: str = "anomaly_label"
    feature_pipeline: str = "identity"
    max_retries: int = 2

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
        label_field="iforest_label",
        feature_pipeline="unit_norm",
        max_retries=3,
    ),
    ModelConfig(
        name="mlp_classifier",
        sensor_type="accel_gyro",
        model_path=_resolve_path(
            "models/mlp_classifier.onnx",
            "models/mlp_classifier.pth",
            "models/mlp_classifier.pt",
        ),
        scaler_path=_resolve_path(
            "models/scaler_mlp.pkl",
            "models/scaler_mlp.joblib",
        ),
        score_field="mlp_score",
        label_field="mlp_label",
        feature_pipeline="identity",
        max_retries=2,
    ),
]


if nn is not None:
    class MLPClassifier(nn.Module):
        def __init__(self, input_size, hidden_sizes=(64, 32), output_size=2):
            super(MLPClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_sizes[0])
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.fc3 = nn.Linear(hidden_sizes[1], output_size)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    class TorchMLPWrapper:
        def __init__(self, model: Any, device: Optional[Any] = None):
            self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            self.model = model.to(self.device)
            self.model.eval()

        def _predict_proba(self, X: np.ndarray) -> np.ndarray:
            if X is None:
                return np.zeros((0, 2))
            with torch.no_grad():
                tx = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
                if tx.dim() == 1:
                    tx = tx.unsqueeze(0)
                out = self.model(tx)
                proba = out.cpu().numpy()
            return proba

        def score_samples(self, X: np.ndarray) -> np.ndarray:
            proba = self._predict_proba(X)
            if proba.size == 0:
                return np.array([])
            mag = np.linalg.norm(proba, axis=1)
            return mag

        def predict(self, X: np.ndarray) -> np.ndarray:
            proba = self._predict_proba(X)
            if proba.size == 0:
                return np.array([])
            binvec = (proba > 0.5).astype(int)
            labels = np.any(binvec, axis=1).astype(int)
            return labels
else:
    class MLPClassifier:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed.")

    class TorchMLPWrapper:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed.")


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

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self._predict_proba(X)
        if proba.size == 0:
            return np.array([])
        binvec = (proba > 0.5).astype(int)
        labels = np.any(binvec, axis=1).astype(int)
        return labels


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
def extract_features(signal: Iterable[float], sampling_rate: float, use_freq_domain: bool = USE_FREQUENCY_DOMAIN) -> list:
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
            if isinstance(loaded_model, dict) and "model_state_dict" in loaded_model:
                try:
                    input_size = loaded_model.get("input_size")
                    hidden_sizes = loaded_model.get("hidden_sizes", (64, 32))
                    output_size = loaded_model.get("output_size", 2)
                    if input_size is None:
                        for k, v in loaded_model["model_state_dict"].items():
                            if k.endswith(".weight") and v is not None:
                                input_size = v.shape[1]
                                break
                    model = MLPClassifier(input_size, hidden_sizes, output_size)
                    try:
                        model.load_state_dict(loaded_model["model_state_dict"])
                    except Exception:
                        pass
                    self.model = TorchMLPWrapper(model)
                except Exception as exc:
                    print(f"Failed to construct Torch MLP from checkpoint: {exc}")
                    self.model = loaded_model
            elif nn is not None and isinstance(loaded_model, nn.Module):
                self.model = TorchMLPWrapper(loaded_model)
            else:
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

    def _run_once(self, window_msg: WindowMessage, features: np.ndarray) -> InferenceResultMessage:
        prepared = self._prepare_features(features)
        score = None
        label = None
        if self.model is not None:
            try:
                score = float(self.model.score_samples(prepared)[0])
            except Exception as exc:
                print(f"Model score error ({self.config.name}):", exc)
            try:
                label = int(self.model.predict(prepared)[0])
            except Exception as exc:
                print(f"Model predict error ({self.config.name}):", exc)
        context_payload: Dict[str, Any] = copy.deepcopy(window_msg.context_payload) if isinstance(window_msg.context_payload, dict) else {}
        context_fields = context_payload.setdefault("fields", {})
        if score is not None:
            context_fields[self.config.score_field] = score
        if label is not None:
            context_fields[self.config.label_field] = label
        context_payload.setdefault("timestamp_ns", window_msg.timestamp_ns or current_timestamp_ns())
        return InferenceResultMessage(
            sensor_type=window_msg.sensor_type,
            score=score,
            label=label,
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
            label=None,
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
        feats = []
        for field in SENSOR_FIELDS:
            sig = window_msg.window_fields.get(field)
            # [확인] 리스트가 들어와도 numpy 배열로 변환합니다.
            arr = np.asarray(sig, dtype=float) if sig is not None else np.asarray([])
            if arr.size < 2:
                feat_len = 11 if USE_FREQUENCY_DOMAIN else 5
                feats.extend([0.0] * feat_len)
            else:
                feats.extend(extract_features(arr, window_msg.sampling_rate_hz, USE_FREQUENCY_DOMAIN))
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