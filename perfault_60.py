import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
techniques=['LOF','Isolation Forest','Autoencoder','LSTM']
spike={'LOF':0.805,'Isolation Forest':0.768,'Autoencoder':0.723,'LSTM':0.293}
drop ={'LOF':1.0,'Isolation Forest':1.0,'Autoencoder':1.0,'LSTM':1.0}
drift={'LOF':0.264,'Isolation Forest':0.265,'Autoencoder':0.255,'LSTM':0.112}
faults=['Spike','Drop','Drift']; x=np.arange(3); w=0.2
colors={'LOF':'#f59e0b','Isolation Forest':'#16a34a','Autoencoder':'#2563eb','LSTM':'#dc2626'}
fig,ax=plt.subplots(figsize=(10,5.5))
for i,t in enumerate(techniques):
    ax.bar(x+(i-1.5)*w,[spike[t],drop[t],drift[t]],w,label=t,color=colors[t])
ax.set_xticks(x); ax.set_xticklabels(faults,fontsize=12)
ax.set_ylabel('Recall (fraction of faults detected)',fontsize=11)
ax.set_title('Detection Recall by Fault Type (pooled across 60 devices)',fontsize=13)
ax.set_ylim(0,1.05); ax.legend(fontsize=10,ncol=2); ax.grid(axis='y',alpha=0.3)
ax.axhline(0,color='black',linewidth=0.8)
plt.tight_layout(); plt.savefig(r'C:/Users/pc/OneDrive/Documents/Desktop/kaggle/fig_perfault_60.png',dpi=130)
print("saved")