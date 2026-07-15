#!/usr/bin/env python3
"""
Futuristic Sensor Dashboard – Light Theme + Alerts + Email
===========================================================
- Model‑based anomaly detection (Tukey's fences + Mahalanobis)
- Real‑time Firebase data, animated gauges, glow effects
- Audio beep on alert, email notifications (once per condition)
- Single Python file – no external HTML/CSS/JS.
"""

import math
import threading
import time
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

# ---------- Firebase (optional) ----------
try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# ---------- Timezone ----------
IST = timezone(timedelta(hours=5, minutes=30))

# ---------- Model: load dataset and compute statistics ----------
def build_model_from_dataset(csv_file='dataset.csv'):
    """Build anomaly detection model from dataset."""
    try:
        df = pd.read_csv(csv_file)
        required = ['temperature', 'humidity', 'voltage', 'current', 'vibration']
        for col in required:
            if col not in df.columns:
                df[col] = 0
        df = df[required]
        df = df[(df != 0).any(axis=1)]
        if df.empty:
            raise ValueError("Dataset empty after cleaning.")

        sensor_stats = {}
        for col in required:
            series = df[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            # Clamp to physical limits
            if col == 'temperature':
                lower, upper = max(lower, 0), min(upper, 60)
            elif col == 'humidity':
                lower, upper = max(lower, 0), min(upper, 100)
            elif col == 'voltage':
                lower, upper = max(lower, 100), min(upper, 300)
            elif col == 'current':
                lower, upper = max(lower, -5), min(upper, 25)
            elif col == 'vibration':
                lower, upper = max(lower, 0), min(upper, 5)

            sensor_stats[col] = {
                'mean': series.mean(),
                'std': series.std(),
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower': lower,
                'upper': upper
            }

        data_matrix = df[required].values
        mean_vec = data_matrix.mean(axis=0)
        cov_matrix = np.cov(data_matrix, rowvar=False)
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

        model = {
            'sensor_stats': sensor_stats,
            'mean_vec': mean_vec,
            'cov_matrix': cov_matrix,
            'features': required,
            'chi2_threshold': chi2.ppf(0.99, df=len(required))
        }
        return model
    except Exception as e:
        print(f"Model build error: {e}. Using fallback fixed ranges.")
        fixed_ranges = {
            'temperature': (20, 40),
            'humidity': (30, 80),
            'voltage': (210, 250),
            'current': (0, 15),
            'vibration': (0, 2.0)
        }
        sensor_stats = {}
        for col in fixed_ranges:
            sensor_stats[col] = {
                'mean': (fixed_ranges[col][0] + fixed_ranges[col][1]) / 2,
                'std': (fixed_ranges[col][1] - fixed_ranges[col][0]) / 4,
                'q1': fixed_ranges[col][0],
                'q3': fixed_ranges[col][1],
                'iqr': fixed_ranges[col][1] - fixed_ranges[col][0],
                'lower': fixed_ranges[col][0],
                'upper': fixed_ranges[col][1]
            }
        mean_vec = np.array([sensor_stats[c]['mean'] for c in fixed_ranges])
        cov_matrix = np.eye(len(fixed_ranges)) * 0.1
        return {
            'sensor_stats': sensor_stats,
            'mean_vec': mean_vec,
            'cov_matrix': cov_matrix,
            'features': list(fixed_ranges.keys()),
            'chi2_threshold': chi2.ppf(0.99, df=len(fixed_ranges))
        }

MODEL = build_model_from_dataset('dataset.csv')

SENSORS = {
    'temperature': {'label': 'Temperature', 'unit': '°C', 'fmt': '{:.1f}', 'firebase_key': 'dht'},
    'humidity':    {'label': 'Humidity',    'unit': '%',  'fmt': '{:.1f}', 'firebase_key': 'dht'},
    'voltage':     {'label': 'Voltage',     'unit': 'V',  'fmt': '{:.1f}', 'firebase_key': 'voltage'},
    'current':     {'label': 'Current',     'unit': 'A',  'fmt': '{:.2f}', 'firebase_key': 'current'},
    'vibration':   {'label': 'Vibration',   'unit': 'g',  'fmt': '{:.3f}', 'firebase_key': 'mpu'}
}

# ---------- Firebase setup ----------
FIREBASE_OK = False
if FIREBASE_AVAILABLE:
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://project-67b08-default-rtdb.firebaseio.com/'
        })
        FIREBASE_OK = True
    except Exception as e:
        print("Firebase init error:", e)
        FIREBASE_OK = False

# ---------- Fetch latest data ----------
def fetch_latest():
    data = {k: np.nan for k in SENSORS}
    timestamps = {k: None for k in SENSORS}
    datetimes = {k: None for k in SENSORS}
    status = {k: False for k in ['dht','voltage','current','mpu']}
    if not FIREBASE_OK:
        return data, timestamps, datetimes, status
    try:
        ref = db.reference('/machines/machine_01/devices/dht11/latest')
        dht = ref.get()
        if dht:
            data['temperature'] = dht.get('temperature', np.nan)
            data['humidity'] = dht.get('humidity', np.nan)
            timestamps['temperature'] = dht.get('timestamp', None)
            datetimes['temperature'] = dht.get('datetime', None)
            timestamps['humidity'] = dht.get('timestamp', None)
            datetimes['humidity'] = dht.get('datetime', None)
            status['dht'] = True

        ref = db.reference('/machines/machine_01/devices/voltage/latest')
        vol = ref.get()
        if vol:
            data['voltage'] = vol.get('value', np.nan)
            timestamps['voltage'] = vol.get('timestamp', None)
            datetimes['voltage'] = vol.get('datetime', None)
            status['voltage'] = True

        ref = db.reference('/machines/machine_01/devices/current/latest')
        cur = ref.get()
        if cur:
            data['current'] = cur.get('value', np.nan)
            timestamps['current'] = cur.get('timestamp', None)
            datetimes['current'] = cur.get('datetime', None)
            status['current'] = True

        ref = db.reference('/machines/machine_01/devices/mpu6050/latest')
        mpu = ref.get()
        if mpu:
            data['vibration'] = mpu.get('value', np.nan)
            timestamps['vibration'] = mpu.get('timestamp', None)
            datetimes['vibration'] = mpu.get('datetime', None)
            status['mpu'] = True
    except Exception as e:
        print("Fetch error:", e)
    return data, timestamps, datetimes, status

# ---------- Format timestamp to IST ----------
def format_timestamp_for_display(ts_ms, dt_str):
    if dt_str is not None and dt_str != "" and not pd.isna(dt_str):
        try:
            dt_utc = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            dt_ist = dt_utc.astimezone(IST)
            return dt_ist.strftime("%H:%M:%S")
        except:
            pass
    if ts_ms is not None and not pd.isna(ts_ms):
        try:
            dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            dt_ist = dt_utc.astimezone(IST)
            return dt_ist.strftime("%H:%M:%S")
        except:
            pass
    return "—"

# ---------- Alert detection using model ----------
def detect_anomalies(values):
    per_sensor = {}
    sensor_stats = MODEL['sensor_stats']
    features = MODEL['features']

    for key in features:
        val = values.get(key, np.nan)
        if pd.isna(val):
            continue
        lower = sensor_stats[key]['lower']
        upper = sensor_stats[key]['upper']
        if val < lower:
            per_sensor[key] = f"{key} too low ({val:.2f} < {lower:.2f})"
        elif val > upper:
            per_sensor[key] = f"{key} too high ({val:.2f} > {upper:.2f})"

    global_anomaly = False
    anomaly_score = 0.0
    vec = []
    valid = True
    for f in features:
        v = values.get(f, np.nan)
        if pd.isna(v):
            valid = False
            break
        vec.append(v)
    if valid:
        vec = np.array(vec)
        mean_vec = MODEL['mean_vec']
        cov_inv = np.linalg.inv(MODEL['cov_matrix'])
        dist = mahalanobis(vec, mean_vec, cov_inv)
        anomaly_score = dist
        if dist > MODEL['chi2_threshold']:
            global_anomaly = True
            if not per_sensor:
                per_sensor['system'] = f"Multivariate anomaly (distance={dist:.2f})"

    return per_sensor, global_anomaly, anomaly_score

# ---------- Normalize value for gauge ----------
def normalize_value(val, key, padding=0.2):
    stats = MODEL['sensor_stats'][key]
    low = stats['lower']
    high = stats['upper']
    span = high - low
    if span <= 0:
        return 0.5
    ext_low = low - span * padding
    ext_high = high + span * padding
    norm = (val - ext_low) / (ext_high - ext_low)
    return max(0.0, min(1.0, norm))

# ---------- Email alert (once per condition) ----------
EMAIL_ENABLED = False
if 'EMAIL_USER' in os.environ and 'EMAIL_PASSWORD' in os.environ and 'EMAIL_TO' in os.environ:
    EMAIL_ENABLED = True
    EMAIL_USER = os.environ['EMAIL_USER']
    EMAIL_PASSWORD = os.environ['EMAIL_PASSWORD']
    EMAIL_TO = os.environ['EMAIL_TO']
    print("Email alerts enabled.")
else:
    print("Email alerts disabled (set EMAIL_USER, EMAIL_PASSWORD, EMAIL_TO env vars).")

# Track sent alerts to avoid duplicates
sent_alerts = set()  # stores (sensor_key, alert_message) or 'global'

def send_alert_email(sensor, message, value=None):
    """Send an email alert (non‑blocking)."""
    if not EMAIL_ENABLED:
        return
    try:
        subject = f"⚠️ Sensor Alert: {sensor}"
        body = f"""
Sensor: {sensor}
Alert: {message}
Value: {value if value is not None else 'N/A'}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Email sent for {sensor}")
    except Exception as e:
        print(f"Email error: {e}")

# ---------- Flask app ----------
app = Flask(__name__)

# Global data cache
latest_data = {
    'values': {k: np.nan for k in SENSORS},
    'timestamps': {k: None for k in SENSORS},
    'datetimes': {k: None for k in SENSORS},
    'status': {k: False for k in ['dht','voltage','current','mpu']},
    'per_sensor_alerts': {},
    'global_anomaly': False,
    'anomaly_score': 0.0,
    'online': {k: False for k in SENSORS},
    'stale': {k: False for k in SENSORS},
    'in_alert': {k: False for k in SENSORS},
    'last_update': None
}
data_lock = threading.Lock()

def background_updater():
    while True:
        try:
            values, timestamps, datetimes, status = fetch_latest()
            per_sensor_alerts, global_anomaly, anomaly_score = detect_anomalies(values)
            alert_keys = set(per_sensor_alerts.keys())

            valid_timestamps = [ts for ts in timestamps.values() if ts is not None and not pd.isna(ts)]
            max_ts = max(valid_timestamps) if valid_timestamps else None

            online = {}
            stale = {}
            for key in SENSORS:
                ts = timestamps.get(key, None)
                raw_online = False
                fk = SENSORS[key]['firebase_key']
                if fk == 'dht':
                    raw_online = status['dht']
                elif fk == 'voltage':
                    raw_online = status['voltage']
                elif fk == 'current':
                    raw_online = status['current']
                elif fk == 'mpu':
                    raw_online = status['mpu']

                if raw_online and ts is not None and not pd.isna(ts) and max_ts is not None:
                    age = max_ts - ts
                    if age <= 5000:
                        online[key] = True
                        stale[key] = False
                    else:
                        online[key] = False
                        stale[key] = True
                else:
                    online[key] = False
                    stale[key] = False

            # Update cache
            with data_lock:
                latest_data['values'] = values
                latest_data['timestamps'] = timestamps
                latest_data['datetimes'] = datetimes
                latest_data['status'] = status
                latest_data['per_sensor_alerts'] = per_sensor_alerts
                latest_data['global_anomaly'] = global_anomaly
                latest_data['anomaly_score'] = anomaly_score
                latest_data['online'] = online
                latest_data['stale'] = stale
                latest_data['in_alert'] = {k: (k in alert_keys or global_anomaly) for k in SENSORS}
                latest_data['last_update'] = datetime.now().isoformat()

            # ----- Email alerts (once per condition) -----
            current_alerts = set()
            # Per‑sensor alerts
            for sensor, msg in per_sensor_alerts.items():
                if sensor == 'system':
                    continue
                key = (sensor, msg)
                current_alerts.add(key)
                if key not in sent_alerts:
                    # Send email
                    val = values.get(sensor, None)
                    send_alert_email(sensor, msg, val)
                    sent_alerts.add(key)
            # Global anomaly
            if global_anomaly:
                global_key = ('global', str(anomaly_score))
                current_alerts.add(global_key)
                if global_key not in sent_alerts:
                    send_alert_email('System', f'Global multivariate anomaly (score={anomaly_score:.2f})')
                    sent_alerts.add(global_key)

            # Remove alerts that are no longer active (so they can be re‑sent later)
            to_remove = [k for k in sent_alerts if k not in current_alerts]
            for k in to_remove:
                sent_alerts.remove(k)

        except Exception as e:
            print("Background updater error:", e)
        time.sleep(2)

thread = threading.Thread(target=background_updater, daemon=True)
thread.start()

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def api_data():
    with data_lock:
        response = {
            'values': {},
            'timestamps': {},
            'datetimes': {},
            'online': {},
            'stale': {},
            'in_alert': {},
            'ranges': {},
            'fmt': {},
            'units': {},
            'labels': {},
            'per_sensor_alerts': latest_data['per_sensor_alerts'],
            'global_anomaly': latest_data['global_anomaly'],
            'anomaly_score': latest_data['anomaly_score'],
            'last_update': latest_data.get('last_update')
        }
        for key in SENSORS:
            val = latest_data['values'].get(key, np.nan)
            response['values'][key] = None if pd.isna(val) else float(val)
            response['timestamps'][key] = latest_data['timestamps'].get(key)
            response['datetimes'][key] = latest_data['datetimes'].get(key)
            response['online'][key] = latest_data['online'].get(key, False)
            response['stale'][key] = latest_data['stale'].get(key, False)
            response['in_alert'][key] = latest_data['in_alert'].get(key, False)
            stats = MODEL['sensor_stats'].get(key, {'lower': 0, 'upper': 1})
            response['ranges'][key] = (stats['lower'], stats['upper'])
            response['fmt'][key] = SENSORS[key]['fmt']
            response['units'][key] = SENSORS[key]['unit']
            response['labels'][key] = SENSORS[key]['label']
        return jsonify(response)

# ---------- Embedded HTML with light theme, animations, beep, glow ----------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ Sensor Dashboard</title>
    <style>
        /* ----- Light futuristic theme ----- */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            background: linear-gradient(135deg, #f0f4ff, #e6edf9);
            font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            position: relative;
            overflow-x: hidden;
        }
        /* Animated background gradient (subtle moving) */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(100, 200, 255, 0.08) 0%, transparent 60%),
                        radial-gradient(circle at 80% 50%, rgba(100, 200, 255, 0.05) 0%, transparent 60%);
            z-index: -1;
            animation: bgShift 20s ease-in-out infinite alternate;
        }
        @keyframes bgShift {
            0% { transform: scale(1) rotate(0deg); }
            100% { transform: scale(1.2) rotate(5deg); }
        }

        .header {
            width: 100%;
            max-width: 1200px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            background: rgba(255,255,255,0.6);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.06);
            border: 1px solid rgba(255,255,255,0.8);
            margin-bottom: 30px;
            transition: box-shadow 0.3s;
        }
        .logo-area {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .logo-icon {
            font-size: 2.2rem;
            animation: float 3s ease-in-out infinite;
        }
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-6px); }
        }
        .logo-text {
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #0077be, #00a8cc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header-right {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 4px;
        }
        .update-time {
            color: #2c3e50;
            font-weight: 500;
            font-size: 0.9rem;
        }
        .anomaly-badge {
            background: #ff6b6b;
            color: #fff;
            padding: 2px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            box-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
            animation: glowBadge 1.2s infinite alternate;
        }
        @keyframes glowBadge {
            0% { box-shadow: 0 0 8px rgba(255,107,107,0.3); }
            100% { box-shadow: 0 0 24px rgba(255,107,107,0.7); }
        }
        .status-text {
            font-size: 0.75rem;
            color: #7f8c8d;
        }

        .dashboard {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
            max-width: 1200px;
            width: 100%;
        }

        .card {
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(8px);
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,0.9);
            padding: 20px 18px 18px;
            position: relative;
            transition: all 0.4s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 280px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.04);
        }
        .card.alert {
            border-color: #ff6b6b;
            box-shadow: 0 0 30px rgba(255,107,107,0.3);
            animation: alertGlow 1.2s infinite alternate;
        }
        @keyframes alertGlow {
            0% { box-shadow: 0 0 20px rgba(255,107,107,0.2); }
            100% { box-shadow: 0 0 50px rgba(255,107,107,0.6); }
        }
        .card.stale {
            border-color: #feca57;
            box-shadow: 0 0 20px rgba(254,202,87,0.2);
        }
        .card.offline {
            opacity: 0.5;
            border-color: #bdc3c7;
        }
        .dot {
            position: absolute;
            top: 16px;
            left: 20px;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            display: inline-block;
            transition: background 0.2s;
        }
        .dot.online {
            background: #00b894;
            box-shadow: 0 0 12px #00b894;
        }
        .dot.stale {
            background: #feca57;
            box-shadow: 0 0 12px #feca57;
            animation: blinkAmber 1s infinite alternate;
        }
        @keyframes blinkAmber {
            0% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .dot.offline {
            background: #bdc3c7;
            animation: blinkOffline 1.2s infinite;
        }
        @keyframes blinkOffline {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.2; }
        }

        .card-title {
            font-size: 1rem;
            font-weight: 600;
            color: #2c3e50;
            letter-spacing: 0.3px;
            margin-left: 28px;
            align-self: flex-start;
            text-transform: uppercase;
        }
        .value {
            font-size: 2.6rem;
            font-weight: 700;
            font-family: 'Roboto Mono', monospace;
            color: #0077be;
            letter-spacing: 1px;
        }
        .value.alert {
            color: #ff6b6b;
        }
        .value.stale {
            color: #feca57;
        }
        .value.offline {
            color: #95a5a6;
        }
        .unit {
            font-size: 1rem;
            color: #7f8c8d;
            margin-left: 4px;
        }
        .range-text {
            font-size: 0.7rem;
            color: #bdc3c7;
            margin-top: 2px;
        }
        .status {
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 4px;
        }
        .status.ideal { color: #00b894; }
        .status.alert { color: #ff6b6b; }
        .status.stale { color: #feca57; }
        .status.offline { color: #95a5a6; }
        .timestamp {
            font-size: 0.7rem;
            color: #bdc3c7;
            margin-top: 2px;
        }
        .gauge-container {
            margin-top: 8px;
            width: 100%;
            display: flex;
            justify-content: center;
        }
        canvas {
            width: 140px;
            height: 80px;
            display: block;
        }

        .footer {
            margin-top: 40px;
            padding: 16px 0;
            border-top: 1px solid rgba(0,0,0,0.05);
            width: 100%;
            max-width: 1200px;
            text-align: center;
            color: #95a5a6;
            font-size: 0.8rem;
        }

        /* Responsive */
        @media (max-width: 820px) {
            .dashboard {
                grid-template-columns: repeat(2, 1fr);
                gap: 18px;
            }
            .header {
                flex-direction: column;
                align-items: flex-start;
                gap: 8px;
            }
            .header-right {
                align-items: flex-start;
                width: 100%;
            }
        }
        @media (max-width: 540px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            canvas {
                width: 120px;
                height: 70px;
            }
            .value {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo-area">
            <span class="logo-icon">⚡</span>
            <span class="logo-text">SENSOR DASH</span>
        </div>
        <div class="header-right">
            <div class="update-time" id="globalUpdate">Updating...</div>
            <div id="globalAnomalyBadge" style="display:none;" class="anomaly-badge">⚠ SYSTEM ANOMALY</div>
            <div class="status-text">Live • Model‑based</div>
        </div>
    </header>

    <div class="dashboard" id="dashboard"></div>

    <footer class="footer">
        &copy; 2026 Sensor Systems &bull; Data refreshed every 2s &bull; Anomaly detection: Tukey's fences + Mahalanobis
    </footer>

    <script>
        // ---------- Audio beep ----------
        function playBeep(frequency=800, duration=200, volume=0.3) {
            try {
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioCtx.createOscillator();
                const gainNode = audioCtx.createGain();
                oscillator.type = 'sine';
                oscillator.frequency.value = frequency;
                gainNode.gain.value = volume;
                oscillator.connect(gainNode);
                gainNode.connect(audioCtx.destination);
                oscillator.start();
                oscillator.stop(audioCtx.currentTime + duration / 1000);
            } catch (e) {
                // Audio not supported
            }
        }

        // ---------- Dashboard setup ----------
        const SENSOR_KEYS = ['temperature', 'humidity', 'voltage', 'current', 'vibration'];
        const dashboard = document.getElementById('dashboard');
        const cardElements = {};

        function createCard(key) {
            const card = document.createElement('div');
            card.className = 'card';
            card.id = `card-${key}`;

            const dot = document.createElement('span');
            dot.className = 'dot offline';
            dot.id = `dot-${key}`;
            card.appendChild(dot);

            const title = document.createElement('div');
            title.className = 'card-title';
            title.id = `label-${key}`;
            title.textContent = key.charAt(0).toUpperCase() + key.slice(1);
            card.appendChild(title);

            const valueWrapper = document.createElement('div');
            valueWrapper.style.display = 'flex';
            valueWrapper.style.alignItems = 'baseline';
            valueWrapper.style.gap = '4px';

            const value = document.createElement('span');
            value.className = 'value offline';
            value.id = `value-${key}`;
            value.textContent = '---';
            valueWrapper.appendChild(value);

            const unit = document.createElement('span');
            unit.className = 'unit';
            unit.id = `unit-${key}`;
            unit.textContent = '';
            valueWrapper.appendChild(unit);

            card.appendChild(valueWrapper);

            const range = document.createElement('div');
            range.className = 'range-text';
            range.id = `range-${key}`;
            range.textContent = '';
            card.appendChild(range);

            const status = document.createElement('div');
            status.className = 'status offline';
            status.id = `status-${key}`;
            status.textContent = '⏳ waiting';
            card.appendChild(status);

            const ts = document.createElement('div');
            ts.className = 'timestamp';
            ts.id = `ts-${key}`;
            ts.textContent = 'Last: —';
            card.appendChild(ts);

            const gaugeDiv = document.createElement('div');
            gaugeDiv.className = 'gauge-container';
            const canvas = document.createElement('canvas');
            canvas.width = 140;
            canvas.height = 80;
            canvas.id = `gauge-${key}`;
            gaugeDiv.appendChild(canvas);
            card.appendChild(gaugeDiv);

            dashboard.appendChild(card);
            cardElements[key] = { card, dot, value, unit, range, status, ts, canvas };
        }

        SENSOR_KEYS.forEach(createCard);

        // ---------- Gauge drawing ----------
        function drawGauge(ctx, cx, cy, radius, norm, color, bgColor) {
            const startAngle = Math.PI;
            const endAngle = startAngle + norm * Math.PI;
            ctx.beginPath();
            ctx.arc(cx, cy, radius, startAngle, startAngle + Math.PI);
            ctx.strokeStyle = bgColor || '#dfe6e9';
            ctx.lineWidth = 6;
            ctx.lineCap = 'round';
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(cx, cy, radius, startAngle, endAngle);
            ctx.strokeStyle = color || '#0077be';
            ctx.lineWidth = 6;
            ctx.lineCap = 'round';
            ctx.shadowColor = color || '#0077be';
            ctx.shadowBlur = 10;
            ctx.stroke();
            ctx.shadowBlur = 0;

            const needleAngle = endAngle;
            const nx = cx + radius * Math.cos(needleAngle);
            const ny = cy + radius * Math.sin(needleAngle);
            ctx.beginPath();
            ctx.arc(nx, ny, 5, 0, 2 * Math.PI);
            ctx.fillStyle = color || '#0077be';
            ctx.shadowColor = color || '#0077be';
            ctx.shadowBlur = 14;
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        function updateGauge(key, norm, color, bgColor) {
            const canvas = cardElements[key].canvas;
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            const cx = w / 2;
            const cy = h - 10;
            const radius = Math.min(w, h * 2) / 2.3;
            drawGauge(ctx, cx, cy, radius, norm, color, bgColor);
        }

        // ---------- Alert state tracking for beep ----------
        let previousAlertState = {};

        // ---------- Update UI from API ----------
        async function updateDashboard() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();

                const values = data.values;
                const online = data.online;
                const stale = data.stale;
                const inAlert = data.in_alert;
                const ranges = data.ranges;
                const fmt = data.fmt;
                const units = data.units;
                const labels = data.labels;
                const timestamps = data.timestamps;
                const datetimes = data.datetimes;
                const lastUpdate = data.last_update;
                const globalAnomaly = data.global_anomaly;
                const perSensorAlerts = data.per_sensor_alerts || {};

                // Global update time
                if (lastUpdate) {
                    const d = new Date(lastUpdate);
                    document.getElementById('globalUpdate').textContent = 'Updated: ' + d.toLocaleTimeString('en-IN', { hour12: false });
                }
                const badge = document.getElementById('globalAnomalyBadge');
                if (globalAnomaly) {
                    badge.style.display = 'inline-block';
                } else {
                    badge.style.display = 'none';
                }

                // Check if any alert is new (for beep)
                let newAlert = false;
                SENSOR_KEYS.forEach(key => {
                    const wasAlert = previousAlertState[key] || false;
                    const nowAlert = inAlert[key] || false;
                    if (nowAlert && !wasAlert) {
                        newAlert = true;
                    }
                    previousAlertState[key] = nowAlert;
                });

                if (newAlert) {
                    playBeep(880, 300, 0.4);
                }

                SENSOR_KEYS.forEach(key => {
                    const el = cardElements[key];
                    const val = values[key];
                    const isOnline = online[key] || false;
                    const isStale = stale[key] || false;
                    const isAlert = inAlert[key] || false;

                    const card = el.card;
                    card.classList.remove('alert', 'stale', 'offline');
                    if (isAlert && isOnline) {
                        card.classList.add('alert');
                    } else if (isStale) {
                        card.classList.add('stale');
                    } else if (!isOnline) {
                        card.classList.add('offline');
                    }

                    const dot = el.dot;
                    dot.className = 'dot';
                    if (isOnline && !isStale) {
                        dot.classList.add('online');
                    } else if (isStale) {
                        dot.classList.add('stale');
                    } else {
                        dot.classList.add('offline');
                    }

                    const low = ranges[key] ? ranges[key][0] : 0;
                    const high = ranges[key] ? ranges[key][1] : 1;
                    el.range.textContent = `Range: ${low.toFixed(1)} – ${high.toFixed(1)} ${units[key]}`;
                    el.unit.textContent = units[key];

                    const valEl = el.value;
                    valEl.className = 'value';
                    if (val !== null && !isNaN(val) && isOnline) {
                        const txt = fmt[key].replace('{', '').replace('}', '').replace(':.', '').replace('f', '');
                        let formatted;
                        if (txt.includes('.')) {
                            const decimals = parseInt(txt.split('.')[1] || '0');
                            formatted = Number(val).toFixed(decimals);
                        } else {
                            formatted = Math.round(val);
                        }
                        valEl.textContent = formatted;
                        if (isAlert) {
                            valEl.classList.add('alert');
                        } else {
                            valEl.classList.remove('alert', 'stale', 'offline');
                        }
                    } else if (isStale && val !== null && !isNaN(val)) {
                        const txt = fmt[key].replace('{', '').replace('}', '').replace(':.', '').replace('f', '');
                        let formatted;
                        if (txt.includes('.')) {
                            const decimals = parseInt(txt.split('.')[1] || '0');
                            formatted = Number(val).toFixed(decimals);
                        } else {
                            formatted = Math.round(val);
                        }
                        valEl.textContent = formatted;
                        valEl.classList.add('stale');
                    } else {
                        valEl.textContent = '---';
                        valEl.classList.add('offline');
                    }

                    const statusEl = el.status;
                    statusEl.className = 'status';
                    if (isOnline && !isStale) {
                        if (isAlert) {
                            statusEl.textContent = '⚠ ALERT';
                            statusEl.classList.add('alert');
                        } else {
                            statusEl.textContent = '● NOMINAL';
                            statusEl.classList.add('ideal');
                        }
                    } else if (isStale) {
                        statusEl.textContent = '⚠ STALE';
                        statusEl.classList.add('stale');
                    } else {
                        statusEl.textContent = '✕ OFFLINE';
                        statusEl.classList.add('offline');
                    }

                    const ts = timestamps[key];
                    const dt = datetimes[key];
                    let displayTs = '—';
                    if (dt && dt !== '') {
                        try {
                            const d = new Date(dt.replace(' ', 'T') + 'Z');
                            displayTs = d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
                        } catch(e) {}
                    } else if (ts !== null && !isNaN(ts)) {
                        try {
                            const d = new Date(ts);
                            displayTs = d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
                        } catch(e) {}
                    }
                    el.ts.textContent = `Last: ${displayTs}`;

                    // Gauge
                    let norm = 0;
                    let gaugeColor = '#dfe6e9';
                    let bgColor = '#dfe6e9';
                    if (val !== null && !isNaN(val) && isOnline) {
                        const span = high - low;
                        const extLow = low - span * 0.2;
                        const extHigh = high + span * 0.2;
                        norm = Math.max(0, Math.min(1, (val - extLow) / (extHigh - extLow)));
                        if (isAlert) {
                            gaugeColor = '#ff6b6b';
                            bgColor = '#ff6b6b';
                        } else {
                            gaugeColor = '#0077be';
                            bgColor = '#0077be';
                        }
                    } else {
                        norm = 0;
                        gaugeColor = '#dfe6e9';
                        bgColor = '#dfe6e9';
                    }
                    updateGauge(key, norm, gaugeColor, bgColor);
                });

            } catch (err) {
                console.error('Update error:', err);
            }
        }

        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
"""

# ---------- Run Flask ----------
if __name__ == '__main__':
    print("✨ Starting Futuristic Sensor Dashboard (Light Theme + Alerts)")
    print("Open http://localhost:5000 in your browser.")
    app.run(debug=False, host='0.0.0.0', port=5000)