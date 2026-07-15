"""
data_collection.py – Fetches ALL historical sensor data from Firebase
history nodes and merges them. Includes debug prints.
"""

import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np

# ---------- Firebase config ----------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://project-67b08-default-rtdb.firebaseio.com/'
})

# ---------- Device definitions ----------
DEVICES = {
    'dht11': {
        'path': '/machines/machine_01/devices/dht11/history',
        'columns': ['temperature', 'humidity'],
        'extra': ['datetime']
    },
    'voltage': {
        'path': '/machines/machine_01/devices/voltage/history',
        'columns': ['value'],
        'rename': 'voltage',
        'extra': ['datetime']
    },
    'current': {
        'path': '/machines/machine_01/devices/current/history',
        'columns': ['value'],
        'rename': 'current',
        'extra': ['datetime']
    },
    'mpu6050': {
        'path': '/machines/machine_01/devices/mpu6050/history',
        'columns': ['value'],            # <-- FIX: read the 'value' key
        'rename': 'vibration',           # rename it to 'vibration' in the DataFrame
        'extra': ['datetime']
    }
}

def fetch_history(device_name):
    """Fetch all history entries for a given device and return a DataFrame."""
    ref = db.reference(DEVICES[device_name]['path'])
    try:
        data = ref.get()
        if data is None:
            print(f"   ⚠️ No data at {DEVICES[device_name]['path']}")
            return pd.DataFrame()
        print(f"   ✅ Raw data has {len(data)} entries")
        rows = []
        for ts_str, values in data.items():
            try:
                timestamp_ms = int(ts_str)
            except ValueError:
                continue
            row = {'timestamp_ms': timestamp_ms}
            for col in DEVICES[device_name]['columns']:
                row[col] = values.get(col, np.nan)
            for extra in DEVICES[device_name].get('extra', []):
                row[extra] = values.get(extra, None)
            rows.append(row)
        df = pd.DataFrame(rows)
        # Rename 'value' to device-specific name if defined
        if 'rename' in DEVICES[device_name]:
            new_name = DEVICES[device_name]['rename']
            if 'value' in df.columns:
                df.rename(columns={'value': new_name}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
        # Generate datetime if missing
        if 'datetime' not in df.columns:
            df['datetime'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # Show sample
        print(f"   Sample first 2 rows:\n{df.head(2).to_string()}\n")
        return df
    except Exception as e:
        print(f"❌ Error fetching {device_name}: {e}")
        return pd.DataFrame()

def merge_all_data():
    """Perform a full outer join on timestamp_ms across all devices."""
    dfs = []
    for device in DEVICES:
        print(f"\n📌 Fetching {device}...")
        df = fetch_history(device)
        if not df.empty:
            print(f"   ✓ {len(df)} records fetched for {device}")
            dfs.append(df)
        else:
            print(f"   ✗ No records for {device}")
    if not dfs:
        print("No data fetched!")
        return pd.DataFrame()

    # Start with the first non-empty DataFrame
    merged = dfs[0]
    for i in range(1, len(dfs)):
        right = dfs[i]
        print(f"\n🔄 Merging with device {i} ({right.shape[0]} rows)...")
        merged = pd.merge(merged, right, on='timestamp_ms', how='outer', suffixes=('', f'_right_{i}'))
        # Clean up duplicate columns
        for col in list(merged.columns):
            if col.endswith('_y') or '_right_' in col:
                if col in ['timestamp_y', 'datetime_y']:
                    merged.drop(columns=[col], inplace=True)
        # Fill datetime if multiple
        if 'datetime' not in merged.columns:
            dt_cols = [c for c in merged.columns if c.startswith('datetime')]
            if dt_cols:
                merged['datetime'] = merged[dt_cols[0]]
        print(f"   After merge: {merged.shape[0]} rows")

    # Define final columns
    feature_cols = ['timestamp_ms', 'timestamp', 'datetime',
                    'temperature', 'humidity', 'voltage', 'current', 'vibration']
    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[feature_cols]
    merged = merged.sort_values('timestamp_ms').reset_index(drop=True)
    return merged

def main():
    print("Collecting ALL historical data from Firebase (full outer join)...")
    df = merge_all_data()
    if df.empty:
        print("No data collected. Exiting.")
        return

    csv_file = 'dataset.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Dataset saved to {csv_file} with {len(df)} records.")
    print("Columns:", df.columns.tolist())
    print("Rows with no missing data:", df.dropna().shape[0])
    print("Rows with at least one missing value:", df.shape[0] - df.dropna().shape[0])

if __name__ == '__main__':
    main()