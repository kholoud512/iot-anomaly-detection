import pandas as pd, os
os.chdir(r'C:\Users\pc\OneDrive\Documents\Desktop\kaggle')
df = pd.read_csv('smart_home_iot_dataset.csv')

# binarize label
df['anom'] = (df['anomaly_label'] >= 1).astype(int)

# for smart_plug:power differ between normal and anomaly rows
for dev in ['smart_plug','thermostat','motion_sensor']:
    sub = df[df['device_type']==dev]
    print(f"\n{dev}")
    print(sub.groupby('anom')['power_consumption_w'].agg(['mean','std','count']))