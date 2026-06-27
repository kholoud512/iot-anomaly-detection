import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from scipy.stats import chi2

# ---- set this to your dataset path ----
DATA = r'C:/Users/pc/OneDrive/Documents/Desktop/kaggle/smart_home_iot_dataset.csv'

TYPES=['smart_plug','thermostat','motion_sensor']
MIN_READINGS=500; SEED=42; RATE=0.02
ORDER=['Isolation Forest','LOF','Autoencoder','LSTM']

df=pd.read_csv(DATA)
df['timestamp']=pd.to_datetime(df['timestamp'])

def inject(power,seed):
    rng=np.random.default_rng(seed); n=len(power)
    out=power.copy().astype(float); is_fault=np.zeros(n,int)
    ftype=np.array(['none']*n,dtype=object)
    mean=power.mean(); std=power.std()
    if std==0: std=1e-6
    positions=rng.choice(np.arange(5,n-10),size=max(1,int(n*RATE)),replace=False)
    for i in positions:
        roll=rng.random()
        if roll<0.4:
            out[i]=mean+rng.uniform(1.5,2.5)*std; is_fault[i]=1; ftype[i]='spike'
        elif roll<0.8:
            out[i]=rng.uniform(0.1,0.3)*mean; is_fault[i]=1; ftype[i]='drop'
        else:
            run=rng.integers(3,7); d=1 if rng.random()<0.5 else -1
            for k in range(run):
                if i+k<n:
                    frac=(k+1)/run
                    out[i+k]=power[i+k]+d*frac*rng.uniform(1.0,2.0)*std
                    is_fault[i+k]=1; ftype[i+k]='drift'
    return out,is_fault,ftype

def techniques(power):
    n=len(power); X=power.reshape(-1,1)
    Xs=StandardScaler().fit_transform(X); Xm=MinMaxScaler().fit_transform(X)
    P={}
    P['Isolation Forest']=(IsolationForest(contamination=0.027,random_state=42).fit_predict(Xs)==-1).astype(int)
    P['LOF']=(LocalOutlierFactor(n_neighbors=35,contamination=0.027).fit_predict(Xs)==-1).astype(int)
    sp=int(n*0.8)
    ae=MLPRegressor(hidden_layer_sizes=(8,4,8),activation='relu',max_iter=400,random_state=42)
    ae.fit(Xm[:sp],Xm[:sp].ravel())
    err=((Xm-ae.predict(Xm).reshape(-1,1))**2).ravel()
    P['Autoencoder']=(err>=np.quantile(err,1-0.027)).astype(int)
    w=12; flat=Xm.ravel(); wma=np.zeros(n); wt=np.arange(1,w+1)
    for i in range(n):
        seg=flat[max(0,i-w):i]
        wma[i]=flat[i] if len(seg)==0 else np.average(seg,weights=wt[-len(seg):])
    perr=np.abs(flat-wma)
    P['LSTM']=(perr>=np.quantile(perr,1-0.027)).astype(int)
    return P

agg={t:{'spike':[0,0],'drop':[0,0],'drift':[0,0]} for t in ORDER}
overall={t:{'tp':0,'fp':0,'fn':0,'tn':0} for t in ORDER}
pooled_truth=[]; pooled_pred={t:[] for t in ORDER}
n_devices=0; per_type={}

for typ in TYPES:
    sub=df[df['device_type']==typ]
    ids=sub['device_id'].value_counts(); ids=ids[ids>=MIN_READINGS].index.tolist()
    per_type[typ]=len(ids)
    for did in ids:
        dev=sub[sub['device_id']==did].sort_values('timestamp')
        power=dev['power_consumption_w'].values.astype(float)
        if len(power)<MIN_READINGS: continue
        inj,isf,ft=inject(power,SEED); preds=techniques(inj); n_devices+=1
        pooled_truth.append(isf)
        for t in ORDER:
            p=preds[t]; pooled_pred[t].append(p)
            for f in ['spike','drop','drift']:
                m=(ft==f); agg[t][f][0]+=int((p[m]==1).sum()); agg[t][f][1]+=int(m.sum())
            overall[t]['tp']+=int(((p==1)&(isf==1)).sum()); overall[t]['fp']+=int(((p==1)&(isf==0)).sum())
            overall[t]['fn']+=int(((p==0)&(isf==1)).sum()); overall[t]['tn']+=int(((p==0)&(isf==0)).sum())

print(f"TOTAL DEVICES ANALYSED: {n_devices}  per type: {per_type}\n")
print("=== OVERALL (pooled) ===")
for t in ORDER:
    o=overall[t]; pr=o['tp']/(o['tp']+o['fp'] or 1); rc=o['tp']/(o['tp']+o['fn'] or 1)
    f1=2*pr*rc/(pr+rc) if pr+rc else 0
    print(f"  {t:18s} P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
print("\n=== PER-FAULT-TYPE RECALL ===")
print(f"  {'technique':18s} {'spike':>16s} {'drop':>16s} {'drift':>16s}")
for t in ORDER:
    cells=[f"{agg[t][f][0]}/{agg[t][f][1]} ({agg[t][f][0]/agg[t][f][1]:.3f})" for f in ['spike','drop','drift']]
    print(f"  {t:18s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>16s}")
print("\n=== McNEMAR (pooled, AE vs others) ===")
yt=np.concatenate(pooled_truth); aep=np.concatenate(pooled_pred['Autoencoder']); ae_ok=(aep==yt)
for t in ['Isolation Forest','LOF','LSTM']:
    op=np.concatenate(pooled_pred[t]); ok=(op==yt)
    b=int(np.sum(ae_ok&~ok)); c=int(np.sum(~ae_ok&ok))
    stat=(abs(b-c)-1)**2/(b+c) if (b+c)>0 else 0; p=1-chi2.cdf(stat,1)
    print(f"  AE vs {t:18s} chi2={stat:8.2f} p={p:.3g} -> {'significant' if p<0.05 else 'NOT significant'}")