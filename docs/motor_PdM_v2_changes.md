# Motor PdM v2.0 - RUL 예측 시스템 개선 문서

## 📋 개요

`motor_PdM_v2.py`는 기존 `motor_PdM.py`의 잔존 수명(RUL) 예측 로직을 개선한 버전입니다.

**수정 일자:** 2025-12-02  
**원본 파일:** `motor_PdM.py` (변경 없음)  
**새 파일:** `motor_PdM_v2.py`

---

## 🔍 기존 버전(v1) 문제점 분석

### 1. 2단계 상태 분류의 한계
```python
# 기존 코드
if current_avg_ms >= FAILURE_THRESHOLD_MS:
    health = "CRITICAL"
else:
    health = "NORMAL"
```
- **문제:** 4999ms → NORMAL, 5000ms → CRITICAL로 급격한 상태 전환
- **결과:** 조기 경고 없이 갑자기 위험 상태로 전환

### 2. 노이즈에 민감한 slope 임계값
```python
# 기존 코드
if slope > 0.01:  # 사이클당 0.01ms
```
- **문제:** 0.01ms/cycle은 너무 작은 값으로, 측정 노이즈도 열화로 판단

### 3. 데이터 부족 시 예측
```python
# 기존 코드
if len(trend_buffer) < 10:
    return "CALCULATING", None, 0.0
```
- **문제:** 10개 샘플로는 안정적인 추세 파악이 어려움

### 4. RUL 시간 계산 오류
```python
# 기존 코드
seconds_left = rul_cycles * (avg_ms / 1000.0)
```
- **문제:** 현재 시점의 평균만 사용하여 미래 사이클 시간 증가를 반영하지 않음

### 5. 이상치 미처리
- 극단값이 선형 회귀에 영향을 미쳐 slope 왜곡

---

## ✅ v2.0 개선 사항

### 1. 3단계 건강 상태 분류

| 상태 | 아이콘 | 조건 | 설명 |
|------|--------|------|------|
| NORMAL | 🟢 | < 4812ms | 정상 운영 |
| WARNING | 🟡 | 4812~5041ms | 주의 관찰 필요 (+5%) |
| CRITICAL | 🔴 | ≥ 5041ms | 즉각 조치 필요 (+10%) |
| CALCULATING | ⚪ | 데이터 부족 | 예측 준비 중 |

```python
# 새 코드
BASELINE_MS = 4583
WARNING_THRESHOLD_MS = int(BASELINE_MS * 1.05)   # 4812ms
FAILURE_THRESHOLD_MS = int(BASELINE_MS * 1.10)   # 5041ms
```

### 2. IQR 기반 이상치 필터링

```python
# 새 코드
q1, q3 = np.percentile(y, [25, 75])
iqr = q3 - q1
valid_mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
```

**효과:**
- 극단적인 측정값이 추세 분석에 미치는 영향 최소화
- 더 안정적인 slope 계산

### 3. 노이즈 필터링 개선

```python
# 새 코드
SLOPE_NOISE_THRESHOLD = 0.1  # 사이클당 0.1ms 이상
MIN_SAMPLES_FOR_PREDICTION = 20  # 최소 20개 샘플
```

**변경점:**
- slope 임계값: 0.01 → 0.1 (10배 증가)
- 최소 샘플 수: 10 → 20 (2배 증가)

### 4. RUL 시간 계산 보정

```python
# 기존 코드
seconds_left = rul_cycles * (avg_ms / 1000.0)

# 새 코드
avg_future_ms = (avg_ms + FAILURE_THRESHOLD_MS) / 2.0
seconds_left = rul_cycles * (avg_future_ms / 1000.0)
```

**원리:**
- 사이클 시간이 증가하는 추세를 고려
- 현재 값과 임계값의 평균을 사용하여 더 현실적인 예측

### 5. 추가 진단 정보

MQTT 메시지에 다음 필드 추가:

```json
{
  "baseline_ms": 4583,
  "warning_threshold_ms": 4812,
  "failure_threshold_ms": 5041,
  "deviation_percent": 2.5
}
```

### 6. 개선된 로그 출력

```
# 기존
[MQTT] 🟢 NORMAL | Cycle=100 | Avg=4600ms | 🕒 RUL: 5.2h

# 새 버전
[MQTT] 🟢 NORMAL | Cycle=100 | Avg=4600ms | Trend: ↑0.05ms/cycle | 🕒 RUL: 5.2h
[MQTT] 🟢 NORMAL | Cycle=100 | Avg=4600ms | Trend: →0.02ms/cycle | ✅ Stable (no degradation)
```

---

## 📊 상태 전이 다이어그램

```
                    +5% (4812ms)           +10% (5041ms)
    ┌─────────┐    ───────────>    ┌─────────┐    ───────────>    ┌──────────┐
    │ 🟢      │                    │ 🟡      │                    │ 🔴       │
    │ NORMAL  │                    │ WARNING │                    │ CRITICAL │
    │         │    <───────────    │         │    <───────────    │          │
    └─────────┘                    └─────────┘                    └──────────┘
```

---

## 🔧 설정 파라미터

| 파라미터 | v1 값 | v2 값 | 설명 |
|----------|-------|-------|------|
| `BASELINE_MS` | (없음) | 4583 | 정상 상태 기준값 |
| `WARNING_THRESHOLD_MS` | (없음) | 4812 | 경고 임계값 (+5%) |
| `FAILURE_THRESHOLD_MS` | 5000 | 5041 | 위험 임계값 (+10%) |
| `SLOPE_NOISE_THRESHOLD` | 0.01 | 0.1 | 열화 판단 slope 임계값 |
| `MIN_SAMPLES_FOR_PREDICTION` | 10 | 20 | 예측 시작 최소 샘플 수 |
| `TREND_WINDOW_SIZE` | 50 | 50 | 추세 분석 윈도우 크기 |

---

## 📦 사용 방법

### 기존 버전 사용
```bash
python motor_PdM.py
```

### 새 버전 사용
```bash
python motor_PdM_v2.py
```

---

## ⚠️ 주의사항

1. **베이스라인 조정:** 다른 모터/환경에서는 `BASELINE_MS` 값을 실측하여 조정 필요
2. **임계값 조정:** WARNING/CRITICAL 비율(5%, 10%)은 운영 환경에 맞게 조정 가능
3. **호환성:** GPIO 핀 배치, MQTT 브로커 설정은 동일

---

## 📈 예상 효과

| 지표 | 개선 내용 |
|------|-----------|
| **조기 경고** | WARNING 단계 추가로 예방 정비 시간 확보 |
| **노이즈 내성** | 오탐(False Positive) 감소 |
| **RUL 정확도** | 미래 사이클 시간 반영으로 정확도 향상 |
| **진단 가시성** | 편차 %, 추세 방향 등 추가 정보 제공 |
