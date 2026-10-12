from pathlib import Path
import json,sys,os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
R=Path(os.environ['SWOT_RELEASE_OUTPUT'])
OUT=R/'figures';OUT.mkdir(exist_ok=True)
pool=pd.read_csv(Path(os.environ['SWOT_RELEASE_ERROR_ANALYSIS'])/'pooled_same_observations.csv')
width=pd.read_csv(R/'width_mapping_summary.csv')
obs=pd.read_csv(str(Path(os.environ['SWOT_RELEASE_DATA_ROOT'])/'postprocess_results/tables/observation_diagnostics.csv'),
                usecols=['validation_scheme','primary_cohort','absolute_error','squared_error'])
obs=obs[obs.validation_scheme.eq('wse_stratified') & obs.primary_cohort.astype(str).str.lower().isin(['true','1'])]
obs=obs.sort_values('absolute_error',ascending=False)
assert len(obs)==6152
rank=100*np.arange(1,len(obs)+1)/len(obs)
share=100*obs.squared_error.cumsum().to_numpy()/obs.squared_error.sum()
BLUE='#28678D';GOLD='#BC8D24';GRAY='#707B82';INK='#27323A'
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,
    'axes.titlesize':10,'axes.titleweight':'bold','axes.edgecolor':INK,
    'axes.linewidth':.75,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
    'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(2,2,figsize=(7.05,6.4))
fig.subplots_adjust(left=.105,right=.985,bottom=.095,top=.90,wspace=.32,hspace=.55)
for ax,title in zip(axs.flat,['a  Concentration of squared error','b  NSE after residual screening',
                              'c  RMSE after residual screening','d  Width and extreme-error occurrence']):
    ax.set_title(title,loc='left',pad=12)
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#E7E9EB',linewidth=.65)
ax=axs[0,0]
ax.plot(rank,share,color=BLUE,lw=1.7)
ax.set_xscale('log');ax.set_xlim(.01,100);ax.set_ylim(0,104)
ax.set_xticks([.01,.1,1,10,100],labels=['0.01','0.1','1','10','100'])
ax.set_yticks([0,25,50,75,100])
ax.set_xlabel('Observations ranked by absolute error (%)\nLargest errors first; logarithmic axis')
ax.set_ylabel('Cumulative squared error (%)')
i=61
ax.scatter(rank[i],share[i],s=31,color=GOLD,edgecolor='white',linewidth=.7,zorder=5)
ax.annotate(f'Largest 62 errors (1.01%)\n{share[i]:.1f}% of squared error',
    xy=(rank[i],share[i]),xytext=(.10,51),fontsize=8.5,ha='left',va='center',
    arrowprops={'arrowstyle':'-','color':GRAY,'lw':.8,'connectionstyle':'angle3,angleA=0,angleB=90'})
rules=['baseline','error10','absolute_top1']
labels=['All\n6152 obs.','Exclude >10-fold\n5695 obs.','Exclude top 1%\n6090 obs.']
x=np.arange(3);w=.31
for ax,metric,ylabel in [(axs[0,1],'NSE','Pooled NSE'),(axs[1,0],'RMSE',r'Pooled RMSE ($\mathrm{m^3\,s^{-1}}$)')]:
    for offset,model,color,label in [(-w/2,'hydraulic',BLUE,'Hydraulic model'),(w/2,'empirical',GOLD,'Empirical WSE–Q')]:
        values=[pool[(pool.rule.eq(rule)) & (pool.model.eq(model))].iloc[0][metric] for rule in rules]
        bars=ax.bar(x+offset,values,w,color=color,label=label,zorder=3)
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    value+((.105 if model=='empirical' else .03) if value>=0 else -.035) if metric=='NSE' else value+25,
                    f'{value:.3f}' if metric=='NSE' else f'{value:.0f}',
                    ha='center',va='bottom' if value>=0 else 'top',fontsize=8)
    ax.set_xticks(x,labels);ax.set_ylabel(ylabel)
    if metric=='NSE':
        ax.set_ylim(-.97,1.12);ax.set_yticks([-.75,-.5,0,.5,1]);ax.axhline(0,color=INK,lw=.75)
    else:
        ax.set_ylim(0,1630);ax.set_yticks([0,400,800,1200,1600])
fig.legend(*axs[0,1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.705,1.005),ncol=2,frameon=False,
                fontsize=8,columnspacing=1.1,handlelength=1.2)
ax=axs[1,1]
order=['raw','full','empirical'];colors=[GRAY,BLUE,GOLD]
values=[100*width[width.model.eq(model)].iloc[0].tenfold_rate for model in order]
counts=[int(width[width.model.eq(model)].iloc[0].tenfold_errors) for model in order]
bars=ax.bar(x,values,.6,color=colors,zorder=3)
for bar,value,count in zip(bars,values,counts):
    ax.text(bar.get_x()+bar.get_width()/2,value+.25,f'{value:.2f}%\n({count}/6152)',
            ha='center',va='bottom',fontsize=8.5)
ax.set_xticks(x,['Hydraulic:\nobserved width','Hydraulic:\nfitted width','Empirical:\nWSE only'])
ax.set_ylim(0,12);ax.set_yticks([0,3,6,9,12]);ax.set_ylabel('Errors exceeding a factor of ten (%)')
ax.text(.02,.98,'WSE, slope and fitted parameters held fixed',transform=ax.transAxes,
        va='top',ha='left',fontsize=7.9,color=GRAY)
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/('F05_extreme_errors_width.'+ext),dpi=450,facecolor='white')
plt.close(fig)
print(OUT/'F05_extreme_errors_width.png')
