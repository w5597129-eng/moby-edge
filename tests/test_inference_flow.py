import math
import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from inference_interface import (
    InferenceResultMessage,
    WindowMessage,
    SENSOR_FIELDS,
)
from inference_worker import InferenceEngine


def test_window_message_roundtrip():
    window_fields = {field: [float(i), float(i + 1)] for i, field in enumerate(SENSOR_FIELDS)}
    msg = WindowMessage(
        sensor_type="accel_gyro",
        sampling_rate_hz=25.0,
        window_fields=window_fields,
        timestamp_ns=123456789,
        context_payload={"sensor_type": "accel_gyro", "fields": {"sample": 1}},
    )
    payload = msg.to_payload()
    parsed = WindowMessage.from_payload(payload)
    assert parsed.sensor_type == msg.sensor_type
    assert parsed.sampling_rate_hz == msg.sampling_rate_hz
    assert parsed.window_fields == msg.window_fields
    assert parsed.context_payload == msg.context_payload


def test_inference_engine_process_window_generates_result():
    class FakeScaler:
        def transform(self, X):
            return X * 0.5

    class FakeModel:
        def score_samples(self, X):
            return np.array([np.sum(X)])

        def predict(self, X):
            return np.array([math.copysign(1, np.sum(X))])

    window_fields = {field: [0.1, 0.2, 0.3] for field in SENSOR_FIELDS}
    msg = WindowMessage(
        sensor_type="accel_gyro",
        sampling_rate_hz=32.0,
        window_fields=window_fields,
        timestamp_ns=999,
        context_payload={"sensor_type": "accel_gyro", "fields": {"accel_x": 0.1}},
    )
    engine = InferenceEngine(model=FakeModel(), scaler=FakeScaler())
    result = engine.process_window(msg)
    assert isinstance(result, InferenceResultMessage)
    assert result.sensor_type == "accel_gyro"
    assert result.score is not None
    assert result.label in (-1, 1)
    assert "anomaly_score" in result.context_payload["fields"]
    assert "anomaly_label" in result.context_payload["fields"]
