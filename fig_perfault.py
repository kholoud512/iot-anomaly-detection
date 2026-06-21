import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

# locked per-fault-type recall 
techniques=['Autoencoder','Isolation Forest','LOF','LSTM']
# average recall across devices
spike={'Autoencoder':(0.88+0.76+0.76)/3,'Isolation Forest':(0.76+0.76+0.76)/3,'LOF':(0.76+0.82+0.76)/3,'LSTM':(0.18+0.29+0.41)/3}
drop ={'Autoencoder':1.0,'Isolation Forest':1.0,'LOF':1.0,'LSTM':1.0}
drift={'Autoencoder':(0.24+0.28+0.32)/3,'Isolation Forest':(0.24+0.28+0.28)/3,'LOF':(0.24+0.28+0.24)/3,'LSTM':(0.10+0.12+0.12)/3}

faults=['Spike','Drop','Drift']
x=np.arange(len(faults)); w=0.2
colors={'Autoencoder':'#2563eb','Isolation Forest':'#16a34a','LOF':'#f59e0b','LSTM':'#dc2626'}

fig,ax=plt.subplots(figsize=(10,5.5))
for i,t in enumerate(techniques):
    vals=[spike[t],drop[t],drift[t]]
    ax.bar(x+(i-1.5)*w, vals, w, label=t, color=colors[t])

ax.set_xticks(x); ax.set_xticklabels(faults, fontsize=12)
ax.set_ylabel('Recall (fraction of faults detected)', fontsize=11)
ax.set_title('Detection Recall by Fault Type (averaged across devices)', fontsize=13)
ax.set_ylim(0,1.05); ax.legend(fontsize=10, ncol=2); ax.grid(axis='y',alpha=0.3)
ax.axhline(0,color='black',linewidth=0.8)
plt.tight_layout(); plt.savefig(r'C:/Users/pc/OneDrive/Documents/Desktop/kaggle/fig_perfault.png',dpi=130)
print("saved fig_perfault.png")