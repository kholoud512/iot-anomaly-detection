import pandas as pd, numpy as np, warnings, os
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score

# ---- folder where injected_*.csv live (CHANGE if needed) ----
FOLDER = r'C:/Users/pc/OneDrive/Documents/Desktop/kaggle'

DEVICES=['smart_plug','thermostat','motion_sensor']
COLORS={'smart_plug':'#16a34a','thermostat':'#f59e0b','motion_sensor':'#2563eb'}
def load(d):
    df=pd.read_csv(os.path.join(FOLDER, f'injected_{d}.csv'))
    return df['power_injected'].values, df['is_fault'].values

neighbors=[5,10,15,20,25,30,35,40,50,60,75]
contams=[0.010,0.015,0.020,0.025,0.027,0.030,0.035,0.040,0.050]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5.5))

for d in DEVICES:
    x,y=load(d); xs=StandardScaler().fit_transform(x.reshape(-1,1))
    f1s=[f1_score(y,(LocalOutlierFactor(n_neighbors=k,contamination=0.027).fit_predict(xs)==-1).astype(int),zero_division=0) for k in neighbors]
    ax1.plot(neighbors,f1s,marker='o',color=COLORS[d],label=d)
ax1.axvline(35,color='red',linestyle='--',alpha=0.7)
ax1.text(35,0.05,' chosen = 35',color='red',fontsize=9)
ax1.set_xlabel('LOF n_neighbors'); ax1.set_ylabel('F1-score')
ax1.set_title('(a) Sensitivity to LOF neighbourhood size'); ax1.set_ylim(0,0.8)
ax1.legend(); ax1.grid(alpha=0.3)

for d in DEVICES:
    x,y=load(d); xs=StandardScaler().fit_transform(x.reshape(-1,1))
    f1s=[f1_score(y,(IsolationForest(contamination=c,random_state=42).fit_predict(xs)==-1).astype(int),zero_division=0) for c in contams]
    ax2.plot(contams,f1s,marker='o',color=COLORS[d],label=d)
ax2.axvline(0.027,color='red',linestyle='--',alpha=0.7)
ax2.text(0.027,0.05,' chosen = 0.027\n (realized fault rate)',color='red',fontsize=9)
ax2.set_xlabel('contamination'); ax2.set_ylabel('F1-score')
ax2.set_title('(b) Sensitivity to contamination (Isolation Forest)'); ax2.set_ylim(0,0.8)
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FOLDER,'fig_sensitivity.png'),dpi=130)
print("Saved fig_sensitivity.png to", FOLDER)