"""
generate_data.py
-----------------
Simulates a time series of Intracranial Pressure (ICP) readings, standing in
for real sensor / Shunt monitoring data (which cannot be used here due to
patient privacy). This is SYNTHETIC data, generated from published normal /
abnormal ICP ranges, intended only for building and demonstrating the
analysis + visualization pipeline.

Clinical reference ranges used (adjust as needed for your literature review):
    Normal resting ICP  : 7 - 15 mmHg
    Mild/Moderate raised : 15 - 20 mmHg
    Pathological / danger : > 20 mmHg (sustained elevation is the concerning signal,
                             since brief spikes from coughing/straining are common
                             and not by themselves pathological)

Two kinds of pathological events are injected, mirroring two real failure modes:
    1. "acute_spike"   - a fast, sharp rise (e.g. acute bleed / obstruction)
    2. "shunt_block"   - a slow, sustained climb over tens of minutes,
                          characteristic of a gradually occluding shunt

Output: icp_data.csv with columns [timestamp, icp_mmhg, event_label]
        event_label is the GROUND TRUTH of what was injected (for validation
        only - the analysis script does NOT get to see this column when it
        does its detection, it only sees icp_mmhg).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SEED = 42
START_TIME = datetime(2026, 8, 20, 8, 0, 0)
DURATION_MINUTES = 240          # 4 hours of monitoring
SAMPLE_INTERVAL_SEC = 30        # one reading every 30 seconds (typical bedside monitor)
BASELINE_MEAN = 11.0            # mmHg, mid-normal range
BASELINE_STD = 1.2              # natural physiological variability
SENSOR_NOISE_STD = 0.6          # additional high-frequency sensor noise

rng = np.random.default_rng(SEED)

n_samples = int((DURATION_MINUTES * 60) / SAMPLE_INTERVAL_SEC)
timestamps = [START_TIME + timedelta(seconds=i * SAMPLE_INTERVAL_SEC) for i in range(n_samples)]

# ----------------------------------------------------------------------
# 1. Base physiological signal: slow random-walk drift + baseline noise
# ----------------------------------------------------------------------
drift = np.cumsum(rng.normal(0, 0.03, n_samples))          # slow wander
baseline_noise = rng.normal(0, BASELINE_STD, n_samples)
icp = BASELINE_MEAN + drift + baseline_noise

# ----------------------------------------------------------------------
# 2. Sensor noise (higher frequency, smaller amplitude "jitter")
# ----------------------------------------------------------------------
sensor_noise = rng.normal(0, SENSOR_NOISE_STD, n_samples)
icp += sensor_noise

# occasional brief "artifact" spikes from coughing/movement (NOT pathological,
# these should be filtered out by smoothing rather than triggering an alarm)
n_artifacts = 8
artifact_idx = rng.choice(n_samples, size=n_artifacts, replace=False)
for idx in artifact_idx:
    icp[idx] += rng.uniform(4, 8)   # single-sample transient blip

event_label = np.array(["normal"] * n_samples, dtype=object)

# ----------------------------------------------------------------------
# 3. Inject a "shunt_block" event: slow sustained climb over ~40 minutes,
#    simulating a gradually occluding shunt, followed by partial recovery
#    (e.g. after intervention)
# ----------------------------------------------------------------------
block_start = int(70 * 60 / SAMPLE_INTERVAL_SEC)     # starts at ~70 min
block_len = int(40 * 60 / SAMPLE_INTERVAL_SEC)        # lasts ~40 min
ramp = np.linspace(0, 14, block_len)                  # climbs to ~+14 mmHg over baseline
icp[block_start:block_start + block_len] += ramp
event_label[block_start:block_start + block_len] = "shunt_block"

recovery_len = int(15 * 60 / SAMPLE_INTERVAL_SEC)
recovery = np.linspace(14, 2, recovery_len)
recov_start = block_start + block_len
icp[recov_start:recov_start + recovery_len] += recovery
event_label[recov_start:recov_start + recovery_len] = "recovery"

# ----------------------------------------------------------------------
# 4. Inject an "acute_spike" event: fast sharp rise and fall over ~8 minutes
#    simulating an acute obstruction / bleed
# ----------------------------------------------------------------------
spike_start = int(170 * 60 / SAMPLE_INTERVAL_SEC)
spike_len = int(8 * 60 / SAMPLE_INTERVAL_SEC)
t = np.linspace(0, np.pi, spike_len)
spike_shape = np.sin(t) * 16      # sharp bell-shaped rise to ~+16 mmHg
icp[spike_start:spike_start + spike_len] += spike_shape
event_label[spike_start:spike_start + spike_len] = "acute_spike"

# ----------------------------------------------------------------------
# 5. Clip to a physiologically plausible floor (ICP can't go far below ~0-2)
# ----------------------------------------------------------------------
icp = np.clip(icp, 2, None)

df = pd.DataFrame({
    "timestamp": timestamps,
    "icp_mmhg": np.round(icp, 2),
    "event_label": event_label,   # ground truth, for validation/demo only
})

out_path = "icp_data.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
print(df["event_label"].value_counts())
