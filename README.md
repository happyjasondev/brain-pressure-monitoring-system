# Brain Pressure Monitoring System
A practice program that monitors intracranial pressure, coded in Python.
It processes time-series intracranial pressure (ICP) readings, flags early signs of shunt blockage or acute pressure surges, and shows the result on a live dashboard.
All data is synthetic.

**How it works:**
1. Simulate	`generate_data.py`:	Creates 4 hours of ICP data (480 readings, one every 30s). It has a normal baseline of about 11 mmHg, sensor noise, brief movement artifacts, a slow shunt-blockage climb, and a sharp acute spike.
2. Analyze `analyze.py`:	Smooths the signal, then applies four checks: a sustained-threshold rule (>20 mmHg for 5+ min), a rate-of-change rule, an Isolation Forest anomaly detector, and a 15-minute linear-trend forecast. These combine into a normal, warning, or high_risk label for each reading.
3. Visualize `dashboard.html`:	Plays the data back like a bedside monitor. It shows a color-coded big number (0-15 mmHg = normal, 15-20 mmHg = elevated, >20 mmHg = danger), a trend chart with shaded risk bands, the forecast, which rule fired, and an alert log. A Simulate acute event button injects a live pressure spike.

## 🛠️ Features & Tech Stack
**Language:**
- Python

**Libraries:**
- numpy
- pandas
- From datetime: datetime, timedelta
- json
- From sklearn.ensemble: IsolationForest

**Key Features:** 
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

## 📖 How to View the Website
In a browser, copy and past this URL on the search bar: `https://brain-pressure-monitoring-system.netlify.app/`

**Disclaimer:**
Analysis intentionally only reads icp_mmhg, never event_label - the ground truth column is dropped before any detection logic runs, so the
pipeline is evaluated the same way it would run on a real unlabeled stream.
This program does not constitute a medical diagnosis. Actual clinical interpretation must still be performed by a professional physician.
