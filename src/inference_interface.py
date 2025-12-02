"""Interfaces and message formats for streaming inference pipelines.

This module centralizes the message schema shared between the sensor
publisher and the inference worker.  It defines dataclasses that carry
windowed sensor samples as well as inference results that are published
back to MQTT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

# Fields used when building feature vectors for inference.
SENSOR_FIELDS: List[str] = [
    "fields_pressure_hpa",
    "fields_accel_x",
    "fields_accel_y",
    "fields_accel_z",
    "fields_gyro_x",
    "fields_gyro_y",
    "fields_gyro_z",
]

# Feature extraction configuration shared between the publisher and worker.
USE_FREQUENCY_DOMAIN: bool = True
WINDOW_SIZE: float = 10.0
WINDOW_OVERLAP: float = 5.0
WINDOW_STEP: float = WINDOW_SIZE - WINDOW_OVERLAP

WINDOW_TOPIC_ROOT = "factory/inference/windows"
RESULT_TOPIC_ROOT = "factory/inference/results"

# V17 Feature Count (Legacy mode deprecated)
EXPECTED_FEATURE_COUNT = 15  # accel(9) + gyro(4) + env(2)


def current_timestamp_ns() -> int:
    """Return the current timestamp in nanoseconds (Py3.7 compatible)."""
    try:
        return int(time.time_ns())
    except AttributeError:  # pragma: no cover - Python < 3.7 fallback
        return int(time.time() * 1e9)


def window_topic(sensor_type: str) -> str:
    return f"{WINDOW_TOPIC_ROOT}/{sensor_type}"


def result_topic(sensor_type: str) -> str:
    return f"{RESULT_TOPIC_ROOT}/{sensor_type}"


def model_result_topic(sensor_type: str, model_name: str) -> str:
    """Return the topic used when publishing per-model inference output."""
    return f"{result_topic(sensor_type)}/{model_name}"


@dataclass
class WindowMessage:
    sensor_type: str
    sampling_rate_hz: float
    window_fields: Dict[str, List[float]]
    timestamp_ns: Optional[int] = None
    context_payload: Optional[Dict[str, Any]] = None

    KIND = "sensor_window"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kind": self.KIND,
            "sensor_type": self.sensor_type,
            "sampling_rate_hz": float(self.sampling_rate_hz),
            "window_fields": self.window_fields,
            "timestamp_ns": self.timestamp_ns or current_timestamp_ns(),
            "context_payload": self.context_payload,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "WindowMessage":
        if payload.get("kind") not in (None, cls.KIND):
            raise ValueError("Unsupported payload kind")
        sensor_type = payload.get("sensor_type")
        if not sensor_type:
            raise ValueError("Missing sensor_type")
        sampling_rate = payload.get("sampling_rate_hz")
        if sampling_rate is None:
            raise ValueError("Missing sampling_rate_hz")
        window_fields = payload.get("window_fields") or {}
        if not isinstance(window_fields, dict):
            raise ValueError("window_fields must be a dict")
        timestamp_ns = payload.get("timestamp_ns")
        context = payload.get("context_payload")
        return cls(
            sensor_type=str(sensor_type),
            sampling_rate_hz=float(sampling_rate),
            window_fields=window_fields,
            timestamp_ns=int(timestamp_ns) if timestamp_ns else None,
            context_payload=context,
        )


@dataclass
class InferenceResultMessage:
    sensor_type: str
    score: Optional[float]
    model_name: Optional[str] = None
    timestamp_ns: Optional[int] = None
    context_payload: Optional[Dict[str, Any]] = None

    KIND = "inference_result"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "kind": self.KIND,
            "sensor_type": self.sensor_type,
            "score": self.score,
            "model_name": self.model_name,
            "timestamp_ns": self.timestamp_ns or current_timestamp_ns(),
            "context_payload": self.context_payload,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "InferenceResultMessage":
        if payload.get("kind") not in (None, cls.KIND):
            raise ValueError("Unsupported payload kind")
        sensor_type = payload.get("sensor_type")
        if not sensor_type:
            raise ValueError("Missing sensor_type")
        score = payload.get("score")
        model_name = payload.get("model_name")
        timestamp_ns = payload.get("timestamp_ns")
        context = payload.get("context_payload")
        return cls(
            sensor_type=str(sensor_type),
            score=float(score) if score is not None else None,
            model_name=str(model_name) if model_name else None,
            timestamp_ns=int(timestamp_ns) if timestamp_ns else None,
            context_payload=context,
        )
