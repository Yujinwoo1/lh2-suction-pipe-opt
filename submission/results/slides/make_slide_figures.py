import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=sys.argv[1]; O=sys.argv[2]
plt.rcParams.update({'font.size':11,'savefig.dpi':300,'savefig.facecolor':'white'})
COL={'gpr':'C0','mlp':'C1','random_forest':'C2','polynomial_ridge':'C3','linear':'C4','physics_baseline':'C7'}
NAME={'gpr':'GPR','mlp':'MLP','random_forest':'RF','polynomial_ridge':'Polynomial','linear':'Linear','physics_baseline':'Physics eq.'}

# F1: test error distribution
p=pd.read_csv(f'{R}/reproduction/final_test_predictions.csv')
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
for a,(t,u) in zip(ax,[('Q_leak_W','Q_leak'),('dp_thermal_Pa','dp_thermal')]):
    d=[100*np.abs(p[f'{t}__{m}']/p[f'{t}__actual']-1) for m in ['gpr','mlp']]
    a.boxplot(d,showfliers=False); a.set_xticks([1,2],['GPR','MLP'])
    a.set_ylabel('Absolute error [%]'); a.set_title(f'{u} (test set, n={len(p)})')
    for i,x in enumerate(d): a.text(i+1,np.percentile(x,75),f' mean {x.mean():.3f}%' if x.mean()<0.01 else f' mean {x.mean():.2f}%',va='bottom',fontsize=9)
fig.tight_layout(); fig.savefig(f'{O}/F1_test_error.png'); plt.close(fig)

# F2: learning curves, same colors
s=pd.read_csv(f'{R}/F2_learning_curve_statistics.csv')
fig,ax=plt.subplots(1,2,figsize=(10,4.6))
for a,(t,u,unit) in zip(ax,[('Q_leak_W','Q_leak','W'),('dp_thermal_Pa','dp_thermal','Pa')]):
    g=s[s.target==t]
    hi0=g[g.model!='polynomial_ridge'].validation_rmse_mean.max()*3
    for m,gm in g.groupby('model'):
        if m=='polynomial_ridge': gm=gm[gm.validation_rmse_mean<hi0]
        a.errorbar(gm.n,gm.validation_rmse_mean,yerr=gm.validation_rmse_std,marker='o',ms=4,capsize=2,color=COL[m],label=NAME[m])
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('Training samples n')
    a.set_ylabel(f'Validation RMSE [{unit}]'); a.set_title(u)
    lo=g.validation_rmse_mean.min()/2; hi=g[g.model!='polynomial_ridge'].validation_rmse_mean.max()*3
    a.set_ylim(lo,hi)
h_,l_=ax[0].get_legend_handles_labels(); fig.legend(h_,l_,loc='lower center',ncol=6,fontsize=9,frameon=False,bbox_to_anchor=(0.5,0.04))
fig.text(0.5,0.005,'GPR uses up to 3,000 samples (n≥3,200 not run). Polynomial is off-scale for n≤100.',ha='center',fontsize=8,color='gray')
fig.tight_layout(rect=(0,0.12,1,1)); fig.savefig(f'{O}/F2_learning_curve.png'); plt.close(fig)

# F3: MLP loss (log) + validation R2, best iteration
h=pd.read_csv(f'{R}/F3_mlp_history.csv')
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
for a,(t,u) in zip(ax,[('Q_leak_W','Q_leak'),('dp_thermal_Pa','dp_thermal')]):
    g=h[h.target==t]; a.plot(g.iteration,g.training_loss_log_target,color='C0',label='Training loss')
    a.set_yscale('log'); a.set_xlabel('Iteration'); a.set_ylabel('Training loss (log10 target)'); a.set_title(f'MLP: {u}')
    b=a.twinx(); b.plot(g.iteration,g.internal_validation_r2_log_target,color='C1',label='Validation R²'); b.set_ylabel('Validation R²')
    bi=g.iteration.iloc[g.internal_validation_r2_log_target.values.argmax()]
    a.axvline(bi,color='gray',ls='--',lw=1); a.text(bi,a.get_ylim()[1],f'best={bi} ',va='top',ha='right',fontsize=9,color='gray')
    l1,n1=a.get_legend_handles_labels(); l2,n2=b.get_legend_handles_labels(); a.legend(l1+l2,n1+n2,fontsize=8,loc='center right')
fig.tight_layout(); fig.savefig(f'{O}/F3_mlp_training.png'); plt.close(fig)

# F5: model comparison (dev CV MAPE)
t=pd.read_csv(f'{R}/T1_model_comparison.csv')
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for a,(tg,u) in zip(ax,[('Q_leak_W','Q_leak'),('dp_thermal_Pa','dp_thermal')]):
    g=t[t.target==tg].sort_values('mape_pct_mean')
    a.bar([NAME[m] for m in g.model],g.mape_pct_mean,color=[COL[m] for m in g.model])
    for i,v in enumerate(g.mape_pct_mean): a.text(i,v,f'{v:.2f}' if v>=0.01 else f'{v:.3f}',ha='center',va='bottom',fontsize=8)
    a.set_yscale('log'); a.set_ylabel('CV mean error (MAPE) [%]'); a.set_title(f'{u} (5-fold CV, n=8,000)'); a.tick_params(axis='x',rotation=20)
fig.tight_layout(); fig.savefig(f'{O}/F5_model_comparison.png'); plt.close(fig)

# F6: hydraulic correction effect (final test)
e=pd.read_csv(f'{R}/T_hydraulic_effect.csv'); e=e[e.evaluation_scope=='single final holdout'].set_index('method')
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
lab=['1.8 fixed','GPR corrected']; c=['gray','C0']
v=[e.loc['const','mape_pct'],e.loc['gpr','mape_pct']]
ax[0].bar(lab,v,color=c); [ax[0].text(i,x,f'{x:.2f}%',ha='center',va='bottom') for i,x in enumerate(v)]
ax[0].set_ylim(0,max(v)*1.2); ax[0].set_ylabel('dp_design error (MAPE) [%]'); ax[0].set_title('Total pressure error (test, n=2,000)')
x=np.arange(2); w=0.35
m1=[e.loc['const','boundary_mismatch_count'],e.loc['gpr','boundary_mismatch_count']]
m2=[e.loc['const','false_safe_count'],e.loc['gpr','false_safe_count']]
ax[1].bar(x-w/2,m1,w,color=c,label='Constraint mismatch'); ax[1].bar(x+w/2,m2,w,color=c,hatch='//',edgecolor='white',label='False safe')
for i in range(2): ax[1].text(x[i]-w/2,m1[i],int(m1[i]),ha='center',va='bottom'); ax[1].text(x[i]+w/2,m2[i],int(m2[i]),ha='center',va='bottom')
ax[1].set_xticks(x,lab); ax[1].set_ylabel('Count'); ax[1].set_title(f'Near 48.3 kPa (n={int(e.loc["const","boundary_n"])})'); ax[1].legend(fontsize=8); ax[1].set_ylim(0,max(m1)+1.5)
fig.tight_layout(); fig.savefig(f'{O}/F6_hydraulic_effect.png'); plt.close(fig)
