import pandas as pd
import os

os.chdir(r'C:\Users\pc\OneDrive\Documents\Desktop\kaggle')

df = pd.read_csv('smart_home_iot_dataset.csv')

print("Shape:", df.shape)
print("\nAll 50 column names:")
print(list(df.columns))
print("\nDevice types available:")
print(df['device_type'].value_counts())
print("\nFirst 3 rows:")
print(df.head(3))