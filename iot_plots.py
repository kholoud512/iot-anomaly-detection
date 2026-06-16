import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import os

OUT = r'C:/Users/pc/OneDrive/Documents/Desktop/kaggle'
DEVICES = ['smart_plug', 'thermostat', 'motion_sensor']

# FIGURE 1: injected signal with faults marked 
fig, axes = plt.subplots(3, 1, figsize=(13, 9))
fig.suptitle('Injected Power Signal per IoT Device (faults in red)', fontsize=13)

for ax, dtype in zip(axes, DEVICES):
    df = pd.read_csv(os.path.join(OUT, f'injected_{dtype}.csv'))
    power = df['power_injected'].values
    fault = df['is_fault'].values == 1
    x = np.arange(len(power))
    ax.plot(x, power, color='#2563eb', linewidth=0.5, alpha=0.7, label='power')
    ax.scatter(x[fault], power[fault], color='#dc2626', s=12, zorder=5,
               label='injected fault')
    ax.set_title(f'{dtype}  (mean {power.mean():.1f} W)')
    ax.set_ylabel('Watts')
    ax.legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel('Reading index (time order)')
plt.tight_layout()
f1path = os.path.join(OUT, 'fig_signals.png')
plt.savefig(f1path, dpi=130)
plt.close()
print('Saved', f1path)

# FIGURE 2: F1 comparison bar chart 
res = pd.read_csv(os.path.join(OUT, 'results_metrics.csv'))
pivot = res.pivot(index='technique', columns='device', values='f1')
pivot = pivot.loc[['Autoencoder', 'Isolation Forest', 'LOF', 'LSTM']]  

fig, ax = plt.subplots(figsize=(10, 5.5))
techs = pivot.index.tolist()
devices = pivot.columns.tolist()
xpos = np.arange(len(techs))
width = 0.25
colors = ['#2563eb', '#16a34a', '#f59e0b']

for i, dev in enumerate(devices):
    vals = pivot[dev].values
    bars = ax.bar(xpos + (i-1)*width, vals, width, label=dev, color=colors[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.2f}',
                ha='center', fontsize=8)

ax.set_xticks(xpos)
ax.set_xticklabels(techs)
ax.set_ylabel('F1-score')
ax.set_ylim(0, 1.0)
ax.set_title('Anomaly Detection F1-score by Technique and IoT Device')
ax.legend(title='Device')
plt.tight_layout()
f2path = os.path.join(OUT, 'fig_f1_comparison.png')
plt.savefig(f2path, dpi=130)
plt.close()
print('Saved', f2path)
print('\nOption A complete: 2 figures generated.')