import pandas as pd
import os

DATA = r'C:/Users/pc/OneDrive/Documents/Desktop/kaggle/smart_home_iot_dataset.csv'
OUT  = r"C:/Users/pc/OneDrive/Documents/Desktop/kaggle"

# the 3 device types we study (high / medium / low power)
TARGETS = ['smart_plug', 'thermostat', 'motion_sensor']

df = pd.read_csv(DATA)
df['timestamp'] = pd.to_datetime(df['timestamp'])

for dtype in TARGETS:
    sub = df[df['device_type'] == dtype]
    # "Device X" = the single device of this type with the most readings
    dev_id = sub['device_id'].value_counts().index[0]
    dev = sub[sub['device_id'] == dev_id].sort_values('timestamp').reset_index(drop=True)

    # time + power
    clean = dev[['timestamp', 'power_consumption_w']].copy()

    fname = os.path.join(OUT, f'clean_{dtype}.csv')
    clean.to_csv(fname, index=False)
    print(f'{dtype}: device {dev_id} -> {len(clean)} readings saved to {fname}')
    print(f'   power mean={clean["power_consumption_w"].mean():.2f}W '
          f'min={clean["power_consumption_w"].min():.2f} '
          f'max={clean["power_consumption_w"].max():.2f}')
print('\nStep 1 complete.')