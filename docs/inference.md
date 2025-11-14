**Inference Summary**
- **File**: `src/sensor_final.py` — async inference worker publishes anomaly results.

**How It Runs**:
- **Trigger**: When the MPU6050 (`accel_gyro`) ring buffer reaches the configured window length, a window is enqueued to the inference queue.
- **Worker**: `inference_worker(client, model, scaler, stop_event)` consumes queue items and performs:
  - Feature extraction: `extract_features()` is called per field in `SENSOR_FIELDS`.
  - Scaling: `scaler.transform(X)` (if scaler loaded).
  - Model inference: `model.score_samples(Xs)` and `model.predict(Xs)` (if model loaded).
  - Payload augmentation: adds `anomaly_score` and `anomaly_label` under `fields`.
  - Publish: sends the augmented payload to `<base_topic>/inference` (e.g. `factory/sensor/accel_gyro/inference`).
  - On publish failure: saves the payload to disk buffer via `save_to_buffer(sensor_type, payload)`.

**Topics**:
- **Sensor topics** (examples):
  - `factory/sensor/accel_gyro`  — raw IMU messages
  - `factory/sensor/vibration`   — vibration
  - `factory/sensor/sound`       — sound
  - `factory/sensor/dht11`       — DHT11
  - `factory/sensor/pressure`    — BMP180
- **Inference topic**: `factory/sensor/<sensor_type>/inference` (constructed in code as `f"{base_topic}/inference"`).

**Payload Shape** (JSON)
- Base sensor payload example (before inference):
```
{
  "sensor_type": "accel_gyro",
  "sensor_model": "MPU6050",
  "fields": {
    "accel_x": 0.123,
    "accel_y": -0.012,
    "accel_z": 0.981,
    "gyro_x": 0.01,
    "gyro_y": -0.02,
    "gyro_z": 0.00
  },
  "timestamp_ns": 1710000000000000000
}
```
- After inference, worker appends fields:
```
{
  "fields": {
    ...,
    "anomaly_score": -0.12345,
    "anomaly_label": -1
  }
}
```
- The full message published to `factory/sensor/accel_gyro/inference` will include the above augmented `fields` and `timestamp_ns`.

**Interpretation**
- `anomaly_label`: model-dependent integer label. For `IsolationForest` used in this repo, typical convention is:
  - `1` => normal
  - `-1` => anomaly
- `anomaly_score`: model's raw score (value and sign meaning depends on model):
  - For `IsolationForest`, lower scores generally indicate more anomalous samples. Confirm with model training notes.
- Consumer systems must document thresholds/logic for converting `anomaly_score` to actionable alerts.

**Timing & Frequency**
- Window size: `WINDOW_SIZE` (seconds) × sampling_rate determines how many IMU samples make one feature vector.
- Enqueue policy: code enqueues a window whenever the ring buffer length >= `buf_len` (sliding window behavior). This can produce frequent inference calls at IMU sampling rate.
- Queue capacity: `inference_q = queue.Queue(maxsize=512)`. If full, the code drops new windows silently (no backpressure). Consider monitoring queue size.

**Failure & Resend Caveats**
- Payloads saved by `save_to_buffer()` are stored as `.npy` files in `BUFFER_DIR` and are re-sent at startup by `resend_buffered(client)`.
- Caveat: `resend_buffered` currently determines the publish topic by `topic_for_type(payload.get("sensor_type"))`. If the saved payload came from an inference publish (which used `.../inference`), resend may publish to the base sensor topic (without `/inference`) unless the payload includes or the code stores the exact publish topic. This can cause inference messages to be re-published to the base topic — consider storing `publish_topic` in saved payloads or sending to `payload.get("publish_topic")` on resend.

**Recommended Improvements**
- Store `publish_topic` with buffered payloads to guarantee correct re-send destination.
- Add an `is_anomaly` boolean derived from `anomaly_score` + threshold to simplify downstream handling.
- Reduce enqueue frequency (e.g., non-overlapping windows or larger step) to lower CPU/ML load.
- Add logging when queue is full instead of silently dropping windows.

**Files / Symbols to review in code**
- `src/sensor_final.py`: `inference_worker`, `load_model_and_scaler`, `save_to_buffer`, `resend_buffered`, `topic_for_type`, main loop where `inference_q.put_nowait(...)` is called.

**Quick Commands**
- Run the sensor publisher (venv recommended):
```
source .venv/bin/activate
python src/sensor_final.py
```
- Re-send buffered files manually (example script can reuse `resend_buffered()` or call `python -c 'from src.sensor_final import resend_buffered, mqtt; ...'` but running the script will auto-resend at startup.)

