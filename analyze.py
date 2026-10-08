"""
analyze.py
----------
Reads icp_data.csv (raw, noisy ICP readings) and produces:
    1. Preprocessed / smoothed signal
    2. Threshold-based early warning (sustained > 20 mmHg for >= 5 minutes)
    3. Rate-of-change early warning (dP/dt too steep)
    4. ML-based anomaly flag (Isolation Forest, unsupervised - does not use
       the ground-truth event_label, only the shape of the signal)
    5. A simple short-horizon trend forecast (next 15 min), using linear
       regression over a trailing window - a lightweight stand-in for an
       LSTM that keeps the demo dependency-free. Swap in a real LSTM
       (e.g. Keras/TensorFlow) for production use; the interface
       (forecast_15min, forecast_slope) stays the same either way.
    6. A combined traffic-light risk_level per timestamp: "normal" / "warning" / "high_risk"

Output: icp_analysis.json - a single JSON file the front-end dashboard reads.
Note: analysis intentionally only reads icp_mmhg, never event_label - the
ground truth column is dropped before any detection logic runs, so the
pipeline is evaluated the same way it would run on a real unlabeled stream.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ----------------------------------------------------------------------
# Config / clinical thresholds
# ----------------------------------------------------------------------
STATIC_THRESHOLD_MMHG = 20.0     # sustained elevation above this is concerning
SUSTAINED_MINUTES = 5            # must persist this long to count as a real warning
RATE_OF_CHANGE_THRESHOLD = 0.35  # mmHg per sample-interval; steep rise flag
SMOOTHING_WINDOW = 5             # rolling mean window (samples)
FORECAST_WINDOW = 20             # samples used to fit trend for forecasting
FORECAST_HORIZON_MIN = 15        # minutes ahead to project

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
df = pd.read_csv("icp_data.csv", parse_dates=["timestamp"])
sample_interval_sec = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()

ground_truth = df["event_label"] if "event_label" in df.columns else None
df = df[["timestamp", "icp_mmhg"]].copy()

# ----------------------------------------------------------------------
# 1. Preprocessing: rolling mean to suppress single-sample sensor artifacts,
#    plus exponential smoothing for a responsive-but-stable trend line
# ----------------------------------------------------------------------
df["icp_smoothed"] = df["icp_mmhg"].rolling(SMOOTHING_WINDOW, center=True, min_periods=1).median()
df["icp_ema"] = df["icp_smoothed"].ewm(span=8, adjust=False).mean()

# ----------------------------------------------------------------------
# 2. Threshold-based warning: smoothed value above threshold, sustained
# ----------------------------------------------------------------------
sustained_samples = int((SUSTAINED_MINUTES * 60) / sample_interval_sec)
above = df["icp_ema"] > STATIC_THRESHOLD_MMHG

# count consecutive True runs
run_id = (above != above.shift()).cumsum()
run_lengths = above.groupby(run_id).transform("size")
df["threshold_warning"] = above & (run_lengths >= sustained_samples)

# ----------------------------------------------------------------------
# 3. Rate-of-change warning: dP/dt on the smoothed signal
# ----------------------------------------------------------------------
df["dP_dt"] = df["icp_ema"].diff().fillna(0)
df["rate_warning"] = df["dP_dt"].abs() > RATE_OF_CHANGE_THRESHOLD

# ----------------------------------------------------------------------
# 4. Isolation Forest anomaly detection (unsupervised)
#    Features: smoothed level + local slope + local volatility
# ----------------------------------------------------------------------
df["rolling_std"] = df["icp_mmhg"].rolling(5, min_periods=1).std().fillna(0)
features = df[["icp_ema", "dP_dt", "rolling_std"]].fillna(0)

iso = IsolationForest(n_estimators=200, contamination=0.12, random_state=42)
df["anomaly_flag"] = iso.fit_predict(features) == -1  # -1 = outlier

# ----------------------------------------------------------------------
# 5. Short-horizon forecast: fit a simple linear trend on the trailing
#    FORECAST_WINDOW samples and project FORECAST_HORIZON_MIN ahead.
#    Recomputed at every timestamp so the dashboard can show a live-updating
#    "predicted trend" line.
# ----------------------------------------------------------------------
horizon_samples = int((FORECAST_HORIZON_MIN * 60) / sample_interval_sec)
forecasts = []
slopes = []
for i in range(len(df)):
    lo = max(0, i - FORECAST_WINDOW + 1)
    window = df["icp_ema"].iloc[lo:i + 1].values
    if len(window) < 3:
        forecasts.append(window[-1] if len(window) else np.nan)
        slopes.append(0.0)
        continue
    x = np.arange(len(window))
    slope, intercept = np.polyfit(x, window, 1)
    projected = intercept + slope * (len(window) - 1 + horizon_samples)
    forecasts.append(round(float(projected), 2))
    slopes.append(round(float(slope), 4))

df["forecast_15min_mmhg"] = forecasts
df["trend_slope"] = slopes

# ----------------------------------------------------------------------
# 6. Combined traffic-light risk level
#    - high_risk : threshold_warning OR (anomaly + forecast above danger)
#    - warning   : rate_warning OR anomaly_flag OR forecast trending into danger
#    - normal    : otherwise
# ----------------------------------------------------------------------
def classify(row):
    if row["threshold_warning"] or row["forecast_15min_mmhg"] > STATIC_THRESHOLD_MMHG + 5:
        return "high_risk"
    if row["rate_warning"] or row["anomaly_flag"] or row["forecast_15min_mmhg"] > STATIC_THRESHOLD_MMHG:
        return "warning"
    return "normal"

df["risk_level"] = df.apply(classify, axis=1)

# ----------------------------------------------------------------------
# Validation against ground truth (only possible because this is
# synthetic demo data - would not exist for a real unlabeled stream)
# ----------------------------------------------------------------------
if ground_truth is not None:
    df["_ground_truth"] = ground_truth
    pathological = ground_truth.isin(["shunt_block", "acute_spike"])
    flagged = df["risk_level"].isin(["warning", "high_risk"])
    tp = (pathological & flagged).sum()
    fn = (pathological & ~flagged).sum()
    fp = (~pathological & flagged).sum()
    print(f"Validation vs injected events -> recall: {tp}/{tp+fn}  "
          f"false-positive samples flagged during 'normal': {fp}")
    df = df.drop(columns=["_ground_truth"])

# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------
records = []
for _, row in df.iterrows():
    records.append({
        "timestamp": row["timestamp"].isoformat(),
        "icp_raw": round(float(row["icp_mmhg"]), 2),
        "icp_smoothed": round(float(row["icp_ema"]), 2),
        "dP_dt": round(float(row["dP_dt"]), 3),
        "forecast_15min_mmhg": row["forecast_15min_mmhg"],
        "trend_slope": row["trend_slope"],
        "threshold_warning": bool(row["threshold_warning"]),
        "rate_warning": bool(row["rate_warning"]),
        "anomaly_flag": bool(row["anomaly_flag"]),
        "risk_level": row["risk_level"],
    })

output = {
    "meta": {
        "static_threshold_mmhg": STATIC_THRESHOLD_MMHG,
        "sustained_minutes": SUSTAINED_MINUTES,
        "rate_of_change_threshold": RATE_OF_CHANGE_THRESHOLD,
        "sample_interval_sec": sample_interval_sec,
        "forecast_horizon_min": FORECAST_HORIZON_MIN,
        "normal_range": [7, 15],
    },
    "series": records,
}

with open("icp_analysis.json", "w") as f:
    json.dump(output, f, indent=2)

risk_counts = df["risk_level"].value_counts().to_dict()
print(f"Wrote icp_analysis.json with {len(records)} points")
print("Risk level distribution:", risk_counts)
