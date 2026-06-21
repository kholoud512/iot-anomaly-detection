import pandas as pd
import numpy as np
import os

OUT = r'C:/Users/pc/OneDrive/Documents/Desktop/kaggle'
DEVICES = ['smart_plug', 'thermostat', 'motion_sensor']
FAULT_RATE = 0.02
SEED = 42

def inject(power, rate, seed):
    rng = np.random.default_rng(seed)
    n = len(power)
    out = power.copy().astype(float)
    is_fault = np.zeros(n, dtype=int)
    fault_type = np.array(['none'] * n, dtype=object)   
    mean = power.mean(); std = power.std()
    n_events = int(n * rate)
    positions = rng.choice(np.arange(5, n - 10), size=n_events, replace=False)
    for i in positions:
        roll = rng.random()
        if roll < 0.4:
            out[i] = mean + rng.uniform(1.5, 2.5) * std
            is_fault[i] = 1; fault_type[i] = 'spike'
        elif roll < 0.8:
            out[i] = rng.uniform(0.1, 0.3) * mean
            is_fault[i] = 1; fault_type[i] = 'drop'
        else:
            run = rng.integers(3, 7)
            direction = 1 if rng.random() < 0.5 else -1
            for k in range(run):
                if i + k < n:
                    frac = (k + 1) / run
                    out[i + k] = power[i + k] + direction * frac * rng.uniform(1.0, 2.0) * std
                    is_fault[i + k] = 1; fault_type[i + k] = 'drift'
    return out, is_fault, fault_type

for dtype in DEVICES:
    df = pd.read_csv(os.path.join(OUT, f'clean_{dtype}.csv'))
    power = df['power_consumption_w'].values
    injected, is_fault, fault_type = inject(power, FAULT_RATE, SEED)
    df['power_injected'] = injected
    df['is_fault'] = is_fault
    df['fault_type'] = fault_type          # NEW column saved
    df.to_csv(os.path.join(OUT, f'injected_{dtype}.csv'), index=False)
    print(f'{dtype}: {len(df)} readings, {is_fault.sum()} faults '
          f'(spike={ (fault_type=="spike").sum() }, '
          f'drop={ (fault_type=="drop").sum() }, '
          f'drift={ (fault_type=="drift").sum() })')
print('\nStep 2 (v3) complete.')