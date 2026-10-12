"""Standalone PNG/PDF figures from saved post-processing tables; no downloads."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.ticker import SymmetricalLogLocator, FuncFormatter, LogLocator
import geopandas as gpd
from common import GROUPS, PRIMARY, read_csv, save

COLORS=['#536878','#D47732','#218C82']
MARKERS=['o','s','^']
LABELS=['Before removal','Slope factor >10 removed','Error factor >10 removed (post hoc)']
SCHEMES=['wse_stratified','temporal_forward']
SCHEME_LABEL={'wse_stratified':'WSE-stratified','temporal_forward':'Temporal-forward'}
XLABEL={'flow_group':'Gauge median discharge Q50 (m3/s)','width_group':'SWORD prior node width (m)',
        'event_flow_group':'Observed discharge at overpass (m3/s)'}

def dots(ax,x,row,color,marker='o'):
    val=row['median']
    if not np.isfinite(val):return
    ax.plot(x,val,marker=marker,color=color,ls='',mfc=color if row.get('stations',5)>=5 else 'white',ms=5,clip_on=False)
    if np.isfinite(row.get('ci_low',np.nan)) and np.isfinite(row.get('ci_high',np.nan)):
        ax.vlines(x,row['ci_low'],row['ci_high'],color=color,lw=1.3)
        ax.plot([x,x],[row['ci_low'],row['ci_high']],marker='_',color=color,ls='')

def missing(ax,text='Insufficient observations'):
    ax.text(.5,.5,text,ha='center',va='center',transform=ax.transAxes,color='#666666')

def padded_shared_y(axes):
    lo=min(a.get_ylim()[0] for a in axes);hi=max(a.get_ylim()[1] for a in axes)
    transform=axes[0].yaxis.get_transform()
    bounds=transform.transform(np.array([lo,hi]))
    pad=max(float(bounds[1]-bounds[0])*.06,1e-6)
    new=transform.inverted().transform([bounds[0]-pad,bounds[1]+pad])
    for ax in axes:ax.set_ylim(*new)

def base_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#777777',
        'pdf.fonttype':42,'savefig.facecolor':'white','xtick.labelsize':9,'ytick.labelsize':9})

def plot_all(root,boundary_file):
    base_style()
    root=Path(root);tables=root/'tables';dest=root/'figures';dest.mkdir(exist_ok=True)
    manifest=[]
    def write(fig,name,source,note):
        fig.text(.06,.015,note,fontsize=8,color='#555555',va='bottom')
        for ext in ('png','pdf'):
            fig.savefig(dest/f'{name}.{ext}',dpi=260,bbox_inches='tight')
        manifest.append(dict(figure=name,source_tables=source,note=note))
        plt.close(fig)
    def legend(fig):
        fig.legend([Line2D([],[],color=c,marker=m,ls='') for c,m in zip(COLORS,MARKERS)],LABELS,
                   loc='upper center',bbox_to_anchor=(.5,.935),ncol=3,frameon=False,fontsize=9)
    r=read_csv(tables/'retention.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));fig.subplots_adjust(bottom=.25,top=.82,wspace=.25)
    fig.suptitle('Observation and evaluable-station retention')
    order=r.rule.drop_duplicates().tolist()
    for ax,metric,label in zip(axes,['observation_retention','station_retention'],['Observations retained (%)','Evaluable stations retained (%)']):
        for j,sc in enumerate(SCHEMES):
            d=r[r.validation_scheme.eq(sc)].set_index('rule').reindex(order)
            ax.bar(np.arange(len(order))+(j-.5)*.36,100*d[metric],width=.36,color=COLORS[j],label=SCHEME_LABEL[sc])
        ax.set_xticks(range(len(order)),[x.replace('absolute_top1','Abs. top 1%') for x in order],rotation=35,ha='right')
        ax.set_ylabel(label);ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    write(fig,'01_retention','retention.csv','Each denominator is the baseline-evaluable cohort. Repeated appearances count once per overpass.\nStation retention requires at least one retained split with >=5 pairs (or the configured minimum).')

    s=read_csv(tables/'scale_spatial_performance_summary.csv')
    s=s[s.comparison.eq('primary_three')]
    for sc in SCHEMES:
        fig,axes=plt.subplots(2,2,figsize=(11.8,8));fig.subplots_adjust(bottom=.20,top=.82,hspace=.45,wspace=.25)
        fig.suptitle('Performance before and after removal: '+SCHEME_LABEL[sc]);legend(fig)
        for ri,metric in enumerate(['MdAPE','NSE']):
            for ci,var in enumerate(['flow_group','width_group']):
                ax=axes[ri,ci];labels=GROUPS[var];counts=[]
                for x,g in enumerate(labels):
                    sub=s[s.validation_scheme.eq(sc)&s.metric.eq(metric)&s.group_variable.eq(var)&s.group.eq(g)]
                    counts.append(int(sub.stations.max()) if len(sub) else 0)
                    for j,rule in enumerate(PRIMARY):
                        z=sub[sub.variant.eq(rule)]
                        if len(z):dots(ax,x+(j-1)*.17,z.iloc[0],COLORS[j],MARKERS[j])
                ax.set_xticks(range(len(labels)),[f'{a}\nn={b}' for a,b in zip(labels,counts)])
                ax.set_xlabel(XLABEL[var]);ax.set_ylabel('MdAPE (%)' if metric=='MdAPE' else 'NSE')
                ax.set_yscale('symlog',linthresh=1 if metric=='NSE' else 10);ax.grid(axis='y',alpha=.2)
                if metric=='NSE':ax.axhline(0,color='#999999',lw=.6)
            padded_shared_y(axes[ri])
            if metric=='MdAPE':
                for ax in axes[ri]:
                    ax.set_ylim(bottom=max(0,ax.get_ylim()[0]))
                    low,high=ax.get_ylim()
                    subs=tuple(range(1,10)) if low>0 and high/low<10 else (1,2,5)
                    ax.yaxis.set_major_locator(SymmetricalLogLocator(base=10,linthresh=10,subs=subs))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
        write(fig,'02_scale_'+sc,'common_station_metrics.csv; scale_spatial_performance_summary.csv',
            'Fixed station-repeat units across all three variants, separately for each metric; no model refitting.\nPoints: station medians; bars: 95% station-bootstrap intervals. Hollow: <5 stations. Symmetric-log axes.\nResidual screening uses observed validation Q; differences describe selected-sample performance.')

    occ=read_csv(tables/'occurrence_rates.csv');occ=occ[occ.population.eq('baseline_evaluable')]
    fig,axes=plt.subplots(2,3,figsize=(14,8));fig.subplots_adjust(bottom=.20,top=.82,wspace=.32,hspace=.48)
    fig.suptitle('Occurrence of large prediction errors and slope flags')
    fig.legend([Line2D([],[],color=c,marker=m,ls='') for c,m in zip(COLORS[1:],MARKERS[1:])],
               ['Slope factor >10','Prediction error factor >10'],loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    for ri,sc in enumerate(SCHEMES):
        for ci,var in enumerate(['flow_group','width_group','event_flow_group']):
            ax=axes[ri,ci];labels=GROUPS[var]
            for x,g in enumerate(labels):
                for j,flag in enumerate(['slope10','error10']):
                    d=occ[occ.validation_scheme.eq(sc)&occ.group_variable.eq(var)&occ.group.eq(g)&occ.flag.eq(flag)]
                    if len(d):
                        row=d.iloc[0].copy();row['median']=100*row.rate;row['ci_low']*=100;row['ci_high']*=100
                        dots(ax,x+(j-.5)*.18,row,COLORS[j+1],MARKERS[j+1])
            shown=[s.replace('Recorded zero','Recorded\nzero').replace('(100,1000]','(100,\n1000]') for s in labels]
            ax.set_xticks(range(len(labels)),shown);ax.set_xlabel(XLABEL[var]);ax.set_ylabel('Flagged observations (%)')
            ax.set_title(SCHEME_LABEL[sc]);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    upper=max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:ax.set_ylim(0,min(100,max(5,upper)))
    write(fig,'03_occurrence_by_scale','occurrence_rates.csv','Denominator: all baseline-evaluable overpasses in each group; repeated predictions median-collapsed first.\nBars: 95% intervals from whole-station resampling. Groups with fewer than 5 stations have no intervals.')

    meta=read_csv(tables/'station_metadata.csv')
    proj='+proj=lcc +lat_1=25 +lat_2=47 +lat_0=35 +lon_0=105 +datum=WGS84 +units=m'
    boundary=gpd.read_file(boundary_file)
    if boundary.crs is None:raise ValueError('Boundary file has no CRS')
    boundary=boundary.to_crs(proj)
    geo=gpd.GeoDataFrame(meta,geometry=gpd.points_from_xy(meta.longitude,meta.latitude),crs='EPSG:4326').to_crs(proj)
    gx=geo.geometry.x;gy=geo.geometry.y
    extent=(gx.min()-150000,gx.max()+150000,gy.min()-150000,gy.max()+150000)
    def map_ax(ax):
        boundary.boundary.plot(ax=ax,color='#d5dadd',linewidth=.3)
        ax.scatter(gx,gy,s=5,color='#d1d1d1',zorder=2)
        ax.set_xlim(*extent[:2]);ax.set_ylim(*extent[2:]);ax.set_aspect('equal');ax.set_axis_off()
    common=read_csv(tables/'common_station_metrics.csv')
    common=common[common.comparison.eq('primary_three')]
    for metric in ['MdAPE','NSE']:
        fig,axes=plt.subplots(2,3,figsize=(13,8));fig.subplots_adjust(bottom=.12,top=.87,wspace=.02,hspace=.09,right=.91)
        fig.suptitle(f'Spatial distribution of {metric}: fixed station-repeat support')
        vals=common.loc[common.metric.eq(metric),'value'].dropna()
        norm=SymLogNorm(linthresh=1 if metric=='NSE' else 10,vmin=min(-1,vals.min()) if metric=='NSE' and len(vals) else 0,
                        vmax=max(1,vals.max()) if len(vals) else 1)
        for ri,sc in enumerate(SCHEMES):
            for ci,rule in enumerate(PRIMARY):
                ax=axes[ri,ci];map_ax(ax)
                d=common[common.validation_scheme.eq(sc)&common.variant.eq(rule)&common.metric.eq(metric)].dropna(subset=['value'])
                g=geo[['station_id','geometry']].merge(d[['station_id','value']],on='station_id',validate='one_to_one')
                scat=ax.scatter(g.geometry.x,g.geometry.y,c=g.value,s=14,norm=norm,cmap='viridis',zorder=3)
                ax.set_title(SCHEME_LABEL[sc]+' / '+['Before','Slope screen','Error screen'][ci]+f' (n={len(g)})',fontsize=10)
        cb=fig.colorbar(scat,cax=fig.add_axes([.93,.23,.015,.52]));cb.set_label(metric+' (%)' if metric=='MdAPE' else metric)
        write(fig,'04_spatial_'+metric,'common_station_metrics.csv; station_metadata.csv',
            'Same metric-specific station-repeat cohort in all columns. Color scale is shared across both validation schemes.\nGray: no eligible metric. Local hydrological boundaries; Lambert conformal projection; no spatial interpolation.')

    sr=read_csv(tables/'station_occurrence.csv');sr=sr[sr.population.eq('baseline_evaluable')]
    fig,axes=plt.subplots(1,2,figsize=(11.4,5.6));fig.subplots_adjust(bottom=.14,top=.86,right=.90,wspace=.03)
    fig.suptitle('Station-level occurrence of prediction errors exceeding a factor of 10')
    for ax,sc in zip(axes,SCHEMES):
        map_ax(ax);d=sr[sr.validation_scheme.eq(sc)]
        g=geo[['station_id','geometry']].merge(d,on='station_id',validate='one_to_one')
        sca=ax.scatter(g.geometry.x,g.geometry.y,c=100*g.error10_rate,s=17,cmap='magma_r',norm=Normalize(0,100),zorder=3)
        ax.set_title(SCHEME_LABEL[sc]+f' (n={len(g)} stations)')
    fig.colorbar(sca,cax=fig.add_axes([.92,.25,.018,.48])).set_label('Large-error observations (%)')
    write(fig,'05_spatial_error_occurrence','station_occurrence.csv','Rate = flagged / all held-out overpasses at the station. Gray: outside baseline-evaluable cohort.\nSmall station samples give uncertain rates; see observation counts in the accompanying table.')

    a=read_csv(tables/'adjusted_associations.csv')
    terms=['log_width','log_Q50','log_relative_Q','log_distance']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True);fig.subplots_adjust(left=.17,bottom=.20,top=.83,wspace=.25)
    fig.suptitle('Exploratory associations with errors exceeding a factor of 10')
    for ax,sc in zip(axes,SCHEMES):
        sub=a[a.validation_scheme.eq(sc)&a.cluster.eq('spatial_block')]
        for y,term in enumerate(terms):
            z=sub[sub.term.eq(term)]
            if len(z):
                row=z.iloc[0];ax.plot(row.odds_ratio,y,'o',color=COLORS[0]);ax.hlines(y,row.ci_low,row.ci_high,color=COLORS[0])
        if sub.empty:missing(ax,'Association not estimable\nSee model-support table')
        ax.set_yticks(range(4),['Prior width','Gauge Q50','Event Q / Q50','1 + matching distance'])
        ax.set_xscale('log');ax.axvline(1,color='#aaaaaa',ls='--');ax.set_xlabel('Odds ratio per tenfold increase')
        ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
        ax.set_title(SCHEME_LABEL[sc]);ax.grid(axis='x',alpha=.2)
    axes[1].tick_params(axis='y',labelleft=False)
    write(fig,'06_adjusted_associations','adjusted_associations.csv; association_model_support.csv',
        'Model also includes four geographic quadrants. Intervals use 5-degree spatial-block clustered covariance.\nAssociations are exploratory, not causal effects. Odds ratios are not probability ratios; station-cluster results are saved too.')

    emp=read_csv(tables/'empirical_scale_comparison.csv');emp=emp[emp.rule.eq('baseline')]
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.5));fig.subplots_adjust(bottom=.20,top=.87,hspace=.45,wspace=.25)
    fig.suptitle('Full model versus empirical WSE-Q curve, by river scale')
    for ri,metric in enumerate(['MdAPE','NSE']):
        for ci,var in enumerate(['flow_group','width_group']):
            ax=axes[ri,ci];counts=[]
            for x,label in enumerate(GROUPS[var]):
                z=emp[emp.metric.eq(metric)&emp.group_variable.eq(var)&emp.group.eq(label)]
                counts.append(int(z.stations.iloc[0]) if len(z) else 0)
                if len(z):dots(ax,x,z.iloc[0],COLORS[0])
            ax.set_xticks(range(len(counts)),[f'{l}\nn={n}' for l,n in zip(GROUPS[var],counts)])
            ax.set_ylabel('MdAPE advantage (percentage points)' if metric=='MdAPE' else 'NSE advantage')
            ax.set_xlabel(XLABEL[var]);ax.axhline(0,color='#999999',lw=.8);ax.set_yscale('symlog',linthresh=1)
        padded_shared_y(axes[ri])
    write(fig,'07_empirical_scale_comparison','empirical_scale_comparison.csv',
        'Positive favors the full model. Exact shared observations within each of all 10 WSE assignments.\nDifferences are summarized within station, then across stations; bars are 95% station-bootstrap intervals.')

    sos=read_csv(tables/'sos_paired_summary.csv')
    sos=sos[sos.cohort.eq('high_confidence')&sos.support.eq('pairwise')&sos.rule.eq('baseline')]
    products=['Consensus','MetroMan','MOMMA','SIC4DVar']
    fig,axes=plt.subplots(1,2,figsize=(11.5,5));fig.subplots_adjust(bottom=.23,top=.82,wspace=.3)
    fig.suptitle('Local model versus audited SoS V3: paired station differences')
    for ax,metric in zip(axes,['MdAPE','NSE']):
        for x,product in enumerate(products):
            for j,sc in enumerate(SCHEMES):
                z=sos[sos.validation_scheme.eq(sc)&sos.metric.eq(metric)&sos['product'].eq(product)]
                if len(z):dots(ax,x+(j-.5)*.18,z.iloc[0],COLORS[j],MARKERS[j])
        ax.set_xticks(range(4),products);ax.axhline(0,color='#999999');ax.set_ylabel('Local-model advantage: '+metric)
        ax.set_yscale('symlog',linthresh=1)
        padded_shared_y([ax])
    fig.legend([Line2D([],[],color=COLORS[i],marker=MARKERS[i],ls='') for i in range(2)],
               [SCHEME_LABEL[x] for x in SCHEMES],loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False)
    write(fig,'08_sos_paired_comparison','sos_paired_summary.csv',
        'High-confidence crosswalk; >=4 pairs per station. Each product and the local model use identical observations.\nPositive favors the local model. Error advantage is in percentage points for MdAPE; intervals resample stations.')

    pooled=read_csv(tables/'sos_pooled_metrics.csv')
    pooled=pooled[pooled.cohort.eq('high_confidence')&pooled.support.eq('fixed_four')&pooled.station_policy.eq('observation_only')]
    fig,axes=plt.subplots(1,2,figsize=(12,5));fig.subplots_adjust(bottom=.24,top=.80,wspace=.25)
    fig.suptitle('Fixed four-product support: sensitivity to observation removal')
    for ax,sc in zip(axes,SCHEMES):
        d=pooled[pooled.validation_scheme.eq(sc)]
        for j,model in enumerate(['Local',*products]):
            z=d[d.model.eq(model)].drop_duplicates(['rule','model']).set_index('rule').reindex(PRIMARY)
            ax.plot(range(3),z.NSE,marker=['o','s','^','D','v'][j],label=model)
        ax.set_xticks(range(3),['Before','Slope >10 removed','Error >10 removed'])
        ax.set_ylabel('Pooled NSE');ax.set_title(SCHEME_LABEL[sc]);ax.set_yscale('symlog',linthresh=1);ax.axhline(0,color='#aaaaaa',lw=.6)
    padded_shared_y(axes)
    axes[0].legend(ncol=3,fontsize=8,frameon=False)
    write(fig,'09_sos_screening_sensitivity','sos_pooled_metrics.csv; sos_retention.csv',
        'All five estimates use the same rows. Baseline requires >=4 pairs per station; observation-only removal keeps low-count stations.\nScreens are defined from the local model. Residual-screened scores are post hoc; coverage and a renewed count-rule analysis are saved.')
    save(pd.DataFrame(manifest),dest,'figure_manifest')
    return manifest

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--boundary-file',type=Path,default=Path('inputs/Class2_hydrounits.shp'))
    a=parser.parse_args();plot_all(a.output_dir,a.boundary_file)
