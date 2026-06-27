import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

# Pooled F1 across 60 devices (locked numbers), ordered best -> worst
techniques=['LOF','Isolation Forest','Autoencoder','LSTM']
f1=[0.689,0.682,0.663,0.502]
colors=['#f59e0b','#16a34a','#2563eb','#dc2626']

fig,ax=plt.subplots(figsize=(8.5,5.2))
bars=ax.bar(techniques,f1,color=colors,width=0.6)
for b,v in zip(bars,f1):
    ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.3f}",
            ha='center',va='bottom',fontsize=11,fontweight='bold')
ax.set_ylabel('F1-score',fontsize=11)
ax.set_title('Overall F1-score by Technique (pooled across 60 devices)',fontsize=13)
ax.set_ylim(0,0.85); ax.grid(axis='y',alpha=0.3)
ax.axhline(0,color='black',linewidth=0.8)
plt.tight_layout(); plt.savefig(r'C:/Users/pc/OneDrive/Documents/Desktop/kaggle/fig_f1_60.png',dpi=130)
print("saved fig_f1_60.png")