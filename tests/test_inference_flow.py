import math
import pathlib
import sys
from typing import List, Optional

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from inference_interface import (
    InferenceResultMessage,
    WindowMessage,
    SENSOR_FIELDS,
)
from inference_worker import (
    InferenceEngine,
    ModelConfig,
    ModelRunner,
    register_feature_pipeline,
)


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


def _build_default_window_message() -> WindowMessage:
    window_fields = {field: [0.1, 0.2, 0.3] for field in SENSOR_FIELDS}
    return WindowMessage(
        sensor_type="accel_gyro",
        sampling_rate_hz=32.0,
        window_fields=window_fields,
        timestamp_ns=999,
        context_payload={"sensor_type": "accel_gyro", "fields": {"accel_x": 0.1}},
    )


def _build_model_config(
    name: str,
    feature_pipeline: str = "identity",
    max_retries: int = 1,
    probability_field_names: Optional[List[str]] = None,
) -> ModelConfig:
    return ModelConfig(
        name=name,
        sensor_type="accel_gyro",
        model_path="unused",
        scaler_path=None,
        result_topic_override=f"factory/test/{name}",
        score_field=f"{name}_score",
        feature_pipeline=feature_pipeline,
        max_retries=max_retries,
        probability_field_names=probability_field_names,
    )


def test_inference_engine_process_window_generates_result():
    class FakeScaler:
        def transform(self, X):
            return X * 0.5

    class FakeModel:
        def score_samples(self, X):
            return np.array([np.sum(X)])

    msg = _build_default_window_message()
    runner = ModelRunner(
        config=_build_model_config("fake", feature_pipeline="identity"),
        model=FakeModel(),
        scaler=FakeScaler(),
    )
    engine = InferenceEngine([runner])
    results = engine.process_window(msg)
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, InferenceResultMessage)
    assert result.sensor_type == "accel_gyro"
    assert result.model_name == "fake"
    assert result.score is not None
    fields = result.context_payload["fields"]
    assert "fake_score" in fields


def test_model_runner_applies_feature_pipeline_unit_norm():
    captured = {"norm": None}

    class CaptureModel:
        def score_samples(self, X):
            captured["norm"] = np.linalg.norm(X)
            return np.array([1.0])

    msg = _build_default_window_message()
    runner = ModelRunner(
        config=_build_model_config("pipeline", feature_pipeline="unit_norm"),
        model=CaptureModel(),
    )
    engine = InferenceEngine([runner])
    engine.process_window(msg)
    assert captured["norm"] is not None
    assert math.isclose(captured["norm"], 1.0, rel_tol=1e-6)


def test_model_runner_retry_and_recovers_from_pipeline_failure():
    attempts = {"count": 0}

    @register_feature_pipeline("test_fail_once")
    def _fail_once(features):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("boom")
        return features

    class ConstantModel:
        def score_samples(self, X):
            return np.array([0.5])

    msg = _build_default_window_message()
    runner = ModelRunner(
        config=_build_model_config("retry", feature_pipeline="test_fail_once", max_retries=2),
        model=ConstantModel(),
    )
    engine = InferenceEngine([runner])
    results = engine.process_window(msg)
    result = results[0]
    assert result.score == 0.5
    assert result.context_payload["fields"]["retry_score"] == 0.5


def test_model_runner_reports_error_after_retries():
    msg = _build_default_window_message()
    runner = ModelRunner(
        config=_build_model_config("error_model", feature_pipeline="identity", max_retries=2),
        model=None,
    )

    def always_fail(window_msg, features):
        raise RuntimeError("always boom")

    runner._run_once = always_fail  # type: ignore
    engine = InferenceEngine([runner])
    result = engine.process_window(msg)[0]
    assert result.score is None
    assert "error_model_error" in result.context_payload["fields"]


def test_model_runner_records_probability_fields():
    class ProbaModel:
        def score_samples(self, X):
            return np.array([0.2])

        def predict(self, X):
            return np.array([1])

        def predict_proba(self, X):
            return np.array([[0.25, 0.75]])

    msg = _build_default_window_message()
    runner = ModelRunner(
        config=_build_model_config(
            "proba",
            probability_field_names=["proba_low", "proba_high"],
        ),
        model=ProbaModel(),
    )
    result = InferenceEngine([runner]).process_window(msg)[0]
    fields = result.context_payload["fields"]
    assert math.isclose(fields["proba_low"], 0.25, rel_tol=1e-6)
    assert math.isclose(fields["proba_high"], 0.75, rel_tol=1e-6)


def test_mlp_score_uses_red_probability():
    class MockMLP:
        def score_samples(self, X):
            return np.array([0.0])

        def predict_proba_raw(self, X):
            return np.array([[0.1, 0.2, 0.7, 0.4, 0.5, 0.6]])

    msg = _build_default_window_message()
    runner = ModelRunner(
        config=_build_model_config(
            "mlp_classifier",
            probability_field_names=[
                "mlp_s1_prob_normal",
                "mlp_s1_prob_yellow",
                "mlp_s1_prob_red",
                "mlp_s2_prob_normal",
                "mlp_s2_prob_yellow",
                "mlp_s2_prob_red",
            ],
        ),
        model=MockMLP(),
    )
    result = InferenceEngine([runner]).process_window(msg)[0]
    assert math.isclose(result.score, 0.7, rel_tol=1e-6)
