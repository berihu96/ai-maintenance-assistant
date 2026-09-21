import pandas as pd
import numpy as np

def generate_sensor_data(num_records=1000):
    np.random.seed(42)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=num_records, freq='1s')
    
    # Baseline telemetry
    vibration = np.random.normal(loc=0.5, scale=0.05, size=num_records)
    temperature = np.random.normal(loc=65.0, scale=2.0, size=num_records)
    rpm = np.random.normal(loc=1750, scale=10, size=num_records)
    
    # Degradation trend over last 200 seconds
    degradation = np.linspace(0, 2.5, 200)
    vibration[-200:] += degradation + np.random.normal(0, 0.1, 200)
    temperature[-200:] += degradation * 8 + np.random.normal(0, 0.5, 200)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'vibration_mms': np.round(vibration, 3),
        'temperature_c': np.round(temperature, 2),
        'rpm': np.round(rpm, 1)
    })
    
    df.to_csv('telemetry_data.csv', index=False)
    print("✓ Successfully created 'telemetry_data.csv'!")

if __name__ == "__main__":
    generate_sensor_data()
