import pandas as pd, numpy as np, warnings, os
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import chi2

OUT = r'C:/Users/pc/OneDrive/Documents/Desktop/kaggle'
DEVICES = ['smart_plug','thermostat','motion_sensor']
ORDER = ['Isolation Forest','LOF','Autoencoder','LSTM']

def run_techniques(power):
    n=len(power); X=power.reshape(-1,1)
    Xs=StandardScaler().fit_transform(X)
    Xm=MinMaxScaler().fit_transform(X)
    preds={}
    preds['Isolation Forest']=(IsolationForest(contamination=0.027,random_state=42).fit_predict(Xs)==-1).astype(int)
    preds['LOF']=(LocalOutlierFactor(n_neighbors=35,contamination=0.027).fit_predict(Xs)==-1).astype(int)
    split=int(n*0.8)
    ae=MLPRegressor(hidden_layer_sizes=(8,4,8),activation='relu',max_iter=600,random_state=42)
    ae.fit(Xm[:split],Xm[:split].ravel())
    err=((Xm-ae.predict(Xm).reshape(-1,1))**2).ravel()
    preds['Autoencoder']=(err>=np.quantile(err,1-0.027)).astype(int)
    w=12; wma=np.zeros(n); weights=np.arange(1,w+1); flat=Xm.ravel()
    for i in range(n):
        seg=flat[max(0,i-w):i]
        wma[i]=flat[i] if len(seg)==0 else np.average(seg,weights=weights[-len(seg):])
    perr=np.abs(flat-wma)
    preds['LSTM']=(perr>=np.quantile(perr,1-0.027)).astype(int)
    return preds

def mcnemar(b,c):
    if (b+c)==0: return 0.0,1.0
    stat=(abs(b-c)-1)**2/(b+c)
    return stat, 1-chi2.cdf(stat,1)

print("="*60)
print("1) OVERALL METRICS (precision, recall, F1)")
print("="*60)
preds_all={}; truth_all={}; ftype_all={}
for d in DEVICES:
    df=pd.read_csv(os.path.join(OUT,f'injected_{d}.csv'))
    power=df['power_injected'].values; y=df['is_fault'].values; ft=df['fault_type'].values
    preds=run_techniques(power); preds_all[d]=preds; truth_all[d]=y; ftype_all[d]=ft
    print(f"\n{d}")
    for t in ORDER:
        p=preds[t]
        print(f"  {t:18s} P={precision_score(y,p,zero_division=0):.2f} "
              f"R={recall_score(y,p,zero_division=0):.2f} "
              f"F1={f1_score(y,p,zero_division=0):.2f}")

print("\n"+"="*60)
print("2) PER-FAULT-TYPE RECALL (caught / total)")
print("="*60)
for d in DEVICES:
    ft=ftype_all[d]; print(f"\n{d}")
    print(f"  {'technique':18s} {'spike':>13s} {'drop':>13s} {'drift':>13s}")
    for t in ORDER:
        p=preds_all[d][t]; cells=[]
        for f in ['spike','drop','drift']:
            m=(ft==f); tot=m.sum(); caught=(p[m]==1).sum()
            cells.append(f"{caught}/{tot} ({caught/tot:.2f})" if tot else "0/0")
        print(f"  {t:18s} {cells[0]:>13s} {cells[1]:>13s} {cells[2]:>13s}")

print("\n"+"="*60)
print("3) McNEMAR'S TEST  (Autoencoder vs each other technique)")
print("="*60)
print("Significant (p<0.05) => the two methods differ on the same readings.")
for d in DEVICES:
    print(f"\n{d}")
    ae=preds_all[d]['Autoencoder']; y=truth_all[d]; ae_ok=(ae==y)
    for t in ['Isolation Forest','LOF','LSTM']:
        ok=(preds_all[d][t]==y)
        b=int(np.sum(ae_ok & ~ok)); c=int(np.sum(~ae_ok & ok))
        stat,p=mcnemar(b,c)
        print(f"  AE vs {t:18s} chi2={stat:6.2f}  p={p:.4f}  "
              f"-> {'significant' if p<0.05 else 'NOT significant'}")