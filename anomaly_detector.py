import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def detect_anomalies():
    df = pd.read_csv('telemetry_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    features = ['vibration_mms', 'temperature_c', 'rpm']
    X = df[features]
    
    model = IsolationForest(contamination=0.10, random_state=42)
    df['anomaly'] = model.fit_predict(X)
    df['is_anomaly'] = df['anomaly'].apply(lambda x: True if x == -1 else False)
    
    anomaly_count = df['is_anomaly'].sum()
    print(f"✓ Analysis Complete: Detected {anomaly_count} anomalous data points out of {len(df)} records.")
    
    df.to_csv('flagged_telemetry.csv', index=False)
    print("✓ Saved flagged data to 'flagged_telemetry.csv'")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['vibration_mms'], label='Vibration (mm/s)', color='blue', alpha=0.6)
    anomalies = df[df['is_anomaly']]
    plt.scatter(anomalies['timestamp'], anomalies['vibration_mms'], color='red', label='Anomaly Detected', s=15)
    plt.title('Industrial Motor Telemetry - Anomaly Detection')
    plt.ylabel('Vibration (mm/s)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['temperature_c'], label='Temperature (°C)', color='orange', alpha=0.6)
    plt.scatter(anomalies['timestamp'], anomalies['temperature_c'], color='red', label='Anomaly Detected', s=15)
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('anomaly_plot.png')
    print("✓ Saved anomaly chart to 'anomaly_plot.png'")

if __name__ == "__main__":
    detect_anomalies()
