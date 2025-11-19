"""Standalone inference worker that consumes sensor windows via MQTT."""
from __future__ import annotations

import copy
import json
import os
import pickle
from typing import Any, Dict, Iterable

import joblib
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew
import paho.mqtt.client as mqtt

from inference_interface import (
    InferenceResultMessage,
    WindowMessage,
    SENSOR_FIELDS,
    USE_FREQUENCY_DOMAIN,
    current_timestamp_ns,
    result_topic,
    WINDOW_TOPIC_ROOT,
)

_DEFAULT_MODEL_PATH = "models/isolation_forest.pkl"
_DEFAULT_SCALER_PATH = "models/scaler_if.pkl"
_RESAVED_MODEL_PATH = "models/resaved_isolation_forest.joblib"
_RESAVED_SCALER_PATH = "models/resaved_scaler.joblib"

MODEL_PATH = _RESAVED_MODEL_PATH if os.path.exists(_RESAVED_MODEL_PATH) else _DEFAULT_MODEL_PATH
SCALER_PATH = _RESAVED_SCALER_PATH if os.path.exists(_RESAVED_SCALER_PATH) else _DEFAULT_SCALER_PATH


def _make_mqtt_client(client_id: str) -> mqtt.Client:
    try:
        return mqtt.Client(client_id=client_id)
    except Exception as e:  # pragma: no cover - compatibility fallback
        try:
            return mqtt.Client(client_id=client_id, callback_api_version=1)
        except Exception:
            try:
                return mqtt.Client(client_id=client_id, userdata=None, protocol=mqtt.MQTTv311)
            except Exception:
                raise e


def extract_features(signal: Iterable[float], sampling_rate: float, use_freq_domain: bool = USE_FREQUENCY_DOMAIN) -> list:
    signal = np.asarray(list(signal), dtype=float)
    if signal.size < 2:
        feature_count = 11 if use_freq_domain else 5
        return [0.0] * feature_count

    abs_signal = np.abs(signal)
    max_val = float(np.max(abs_signal))
    abs_mean = float(np.mean(abs_signal))
    std = float(np.std(signal))
    peak_to_peak = float(np.ptp(signal))
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
    spectral_kurt = float(kurtosis(spectrum, fisher=False)) if spectrum.size > 1 else 0.0
    spectral_skewness = float(skew(spectrum)) if spectrum.size > 1 else 0.0
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


def load_model_and_scaler():
    model = None
    scaler = None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        try:
            with open(MODEL_PATH, "rb") as fh:
                model = pickle.load(fh)
        except Exception as exc:
            print("Model load error:", exc)
            model = None
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        try:
            with open(SCALER_PATH, "rb") as fh:
                scaler = pickle.load(fh)
        except Exception as exc:
            print("Scaler load error:", exc)
            scaler = None
    return model, scaler


class InferenceEngine:
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler

    def _build_feature_vector(self, window_msg: WindowMessage) -> np.ndarray:
        feats = []
        for field in SENSOR_FIELDS:
            sig = window_msg.window_fields.get(field)
            if not sig or len(sig) < 2:
                feat_len = 11 if USE_FREQUENCY_DOMAIN else 5
                feats.extend([0.0] * feat_len)
            else:
                feats.extend(extract_features(sig, window_msg.sampling_rate_hz, USE_FREQUENCY_DOMAIN))
        return np.asarray(feats, dtype=float).reshape(1, -1)

    def process_window(self, window_msg: WindowMessage) -> InferenceResultMessage:
        X = self._build_feature_vector(window_msg)
        Xs = X
        if self.scaler is not None:
            try:
                Xs = self.scaler.transform(X)
            except Exception:
                Xs = X
        score = None
        label = None
        if self.model is not None:
            try:
                score = float(self.model.score_samples(Xs)[0])
            except Exception:
                pass
            try:
                label = int(self.model.predict(Xs)[0])
            except Exception:
                pass
        context = window_msg.context_payload or {
            "sensor_type": window_msg.sensor_type,
            "fields": {},
            "timestamp_ns": window_msg.timestamp_ns or current_timestamp_ns(),
        }
        # Copy context to avoid mutating shared dict
        context_payload: Dict[str, Any] = copy.deepcopy(context) if isinstance(context, dict) else {}
        context_fields = context_payload.setdefault("fields", {})
        if score is not None:
            context_fields["anomaly_score"] = score
        if label is not None:
            context_fields["anomaly_label"] = label
        context_payload.setdefault("timestamp_ns", window_msg.timestamp_ns or current_timestamp_ns())
        return InferenceResultMessage(
            sensor_type=window_msg.sensor_type,
            score=score,
            label=label,
            timestamp_ns=current_timestamp_ns(),
            context_payload=context_payload,
        )


class MQTTInferenceWorker:
    def __init__(self, broker: str = "localhost", port: int = 1883):
        self.broker = broker
        self.port = port
        self.engine = InferenceEngine(*load_model_and_scaler())
        self.client = _make_mqtt_client("sensor_inference_worker")
        self.client.on_message = self._on_message
        self.client.on_connect = self._on_connect

    def _on_connect(self, client, userdata, flags, rc):  # pragma: no cover - network callback
        if rc == 0:
            client.subscribe(f"{WINDOW_TOPIC_ROOT}/#")

    def _on_message(self, client, userdata, msg):  # pragma: no cover - network callback
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            window_msg = WindowMessage.from_payload(payload)
            result = self.engine.process_window(window_msg)
            client.publish(result_topic(result.sensor_type), json.dumps(result.to_payload()))
            print(
                f"INFERENCE | {result.sensor_type} | score={result.score} label={result.label}"
            )
        except Exception as exc:
            print("Inference MQTT handler error:", exc)

    def start(self):  # pragma: no cover - network loop
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_forever()


def main():  # pragma: no cover - CLI entry point
    worker = MQTTInferenceWorker()
    worker.start()


if __name__ == "__main__":  # pragma: no cover
    main()
