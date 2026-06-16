import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

OUT = r'C:/Users/pc/OneDrive/Documents/Desktop/kaggle'
DEVICES = ['smart_plug', 'thermostat', 'motion_sensor']
# roughly matches realized fault rate
CONTAM = 0.027   
# single fixed tuning choice, same for all devices
LOF_NEIGHBORS = 35   

def run_isolation_forest(x):
    xs = StandardScaler().fit_transform(x.reshape(-1, 1))
    return (IsolationForest(contamination=CONTAM, random_state=42)
            .fit_predict(xs) == -1).astype(int)

def run_lof(x):
    xs = StandardScaler().fit_transform(x.reshape(-1, 1))
    return (LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, contamination=CONTAM)
            .fit_predict(xs) == -1).astype(int)

def run_autoencoder(x):
    xs = MinMaxScaler().fit_transform(x.reshape(-1, 1))
    ae = MLPRegressor(hidden_layer_sizes=(8, 4, 8), activation='relu',
                      max_iter=300, random_state=42)
    ae.fit(xs, xs.ravel())
    err = np.abs(xs.ravel() - ae.predict(xs))
    return (err > np.quantile(err, 1 - CONTAM)).astype(int)

def run_lstm_style(x):
    xs = MinMaxScaler().fit_transform(x.reshape(-1, 1)).ravel()
    window = 12
    pred = xs.copy()
    w = np.exp(np.linspace(-1, 0, window)); w /= w.sum()
    for i in range(window, len(xs)):
        pred[i] = np.dot(xs[i-window:i], w)
    err = np.abs(xs - pred)
    return (err > np.quantile(err, 1 - CONTAM)).astype(int)

TECHS = {'Isolation Forest': run_isolation_forest,
         'Autoencoder': run_autoencoder,
         'LSTM': run_lstm_style,
         'LOF': run_lof}

rows = []
for dtype in DEVICES:
    df = pd.read_csv(os.path.join(OUT, f'injected_{dtype}.csv'))
    x = df['power_injected'].values
    y = df['is_fault'].values
    print(f'\n========== {dtype} ({len(df)} readings, {y.sum()} faults) ==========')
    print(f'{"Technique":<18}{"Prec":>7}{"Recall":>8}{"F1":>7}   TP  FP  FN   TN')
    print('-'*58)
    for name, fn in TECHS.items():
        pred = fn(x)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f = f1_score(y, pred, zero_division=0)
        tn, fp, fn_, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        print(f'{name:<18}{p:>7.2f}{r:>8.2f}{f:>7.2f}   {tp:>2} {fp:>3} {fn_:>3} {tn:>4}')
        rows.append({'device': dtype, 'technique': name,
                     'precision': round(p,3), 'recall': round(r,3), 'f1': round(f,3)})

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'results_metrics.csv'), index=False)
print('\n\nCOMPARISON: F1 across devices')
print(res.pivot(index='technique', columns='device', values='f1').to_string())