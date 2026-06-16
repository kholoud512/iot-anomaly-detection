import pandas as pd
import os

os.chdir(r'C:\Users\pc\OneDrive\Documents\Desktop\kaggle')
df = pd.read_csv('smart_home_iot_dataset.csv')

print("anomaly_label values")
print(df['anomaly_label'].value_counts())
print("Anomaly rate:", df['anomaly_label'].mean())

print("\navg power_consumption_w per device type")
print(df.groupby('device_type')['power_consumption_w'].agg(['mean','min','max']))

print("\nreadings per device (top 5)")
print(df['device_id'].value_counts().head())

print("\nmissing values in key columns")
print(df[['timestamp','device_id','device_type','power_consumption_w','anomaly_label']].isnull().sum())