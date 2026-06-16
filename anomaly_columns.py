import pandas as pd, os
os.chdir(r'C:\Users\pc\OneDrive\Documents\Desktop\kaggle')
df = pd.read_csv('smart_home_iot_dataset.csv')
df['anom'] = (df['anomaly_label'] >= 1).astype(int)

# compare normal vs anomaly across all candidate columns
cols = ['power_consumption_w','voltage_v','current_a','power_factor','peak_demand_w',
        'rssi_dbm','snr_db','latency_ms','packet_loss_pct','uptime_hours',
        'connection_drops_24h','error_count_24h','sensor_drift_pct',
        'network_anomaly_score','battery_level_pct','temperature_c']

for c in cols:
    if c in df.columns:
        g = df.groupby('anom')[c].mean()
        diff = abs(g[1]-g[0])
        rel = diff/(abs(g[0])+1e-9)
        print(f"{c:25s} normal={g[0]:10.3f}  anom={g[1]:10.3f}  rel_diff={rel:.2%}")