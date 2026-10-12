from pathlib import Path
import sys, importlib.util, json, shutil, os
import xml.etree.ElementTree as ET
import base64,zlib,urllib.parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT=Path(os.environ['SWOT_RELEASE_OUTPUT']).resolve()
OUT=ROOT
for x in ['main','supplement','source','editable','assets']:(OUT/x).mkdir(parents=True,exist_ok=True)
sys.argv=['build_figures.py','--out',str(OUT),'--cache',os.environ['SWOT_RELEASE_DOMAIN_CACHE'],'--pdf-work',str(ROOT/'pdf'),'--boundary',os.environ['SWOT_RELEASE_BOUNDARY']]
spec=importlib.util.spec_from_file_location('oldfig',Path(__file__).with_name('plot_helpers.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
D=old.D;BLUE=old.BLUE;ORANGE=old.ORANGE;TEAL=old.TEAL;GRAY=old.GRAY;INK=old.INK;LIGHT=old.LIGHT
panel=old.panel
original_finish=old.finish
def finish(f,*args,**kwargs):
    from matplotlib.ticker import NullFormatter
    for ax in f.axes:
        if ax.get_xscale()=='log':ax.xaxis.set_minor_formatter(NullFormatter())
        if ax.get_yscale()=='log':ax.yaxis.set_minor_formatter(NullFormatter())
    return original_finish(f,*args,**kwargs)


def support():
    f=plt.figure(figsize=(7.1,6.2));gs=f.add_gridspec(2,2,left=.09,right=.975,bottom=.1,top=.94,hspace=.62,wspace=.42,height_ratios=[1.1,1])
    a=f.add_subplot(gs[0,0]);panel(a,'a','Calibration support across China');old.base_map(a)
    z=D['global_qc_station_summary'];z=z[z.station_id.isin(D['parameters'].station_id)];g=old.xy(z)
    sc=a.scatter(g.geometry.x,g.geometry.y,c=g.retained_observations,s=12,cmap='viridis',norm=old.LogNorm(vmin=5,vmax=g.retained_observations.max()),edgecolor='white',lw=.2)
    cb=f.colorbar(sc,ax=a,orientation='horizontal',fraction=.06,pad=.04);cb.set_label('Retained observations at 542 fitted stations',fontsize=7)
    b=f.add_subplot(gs[0,1]);panel(b,'b','From the archive to usable pairs')
    vals=[78643,10756,8859,8268];labs=['Archived','Complete','After QC','Fitted record'];b.barh(range(4),vals,color=[LIGHT,GRAY,BLUE,TEAL],height=.58);b.invert_yaxis();b.set_xscale('log');b.set_yticks(range(4),labs);b.set_xlabel('Station-overpasses (log axis)');b.grid(False);b.set_xlim(1000,160000)
    for j,v in enumerate(vals):b.text(v*1.08,j,f'{v:,}',va='center',fontsize=7.5)
    c=f.add_subplot(gs[1,0]);c.set_position([.215,.10,.29,.29]);panel(c,'c','Contribution of hydraulic components')
    keys=['width_train_only','no_monotonic','n_only_fixed_geometry','fixed_reference_n','beta_fixed_1','empirical_rating_curve']
    labels=['Measured test width','Observed widths','n only, fixed geometry','Fixed reference n','β fixed at 1','Empirical WSE–Q']
    z=D['ablation_paired_test_summary'].query("metric == 'NSE'").set_index('comparator').loc[keys]
    for i,(key,row) in enumerate(z.iterrows()):
        col=ORANGE if key=='empirical_rating_curve' else BLUE;c.plot([row.ci_lower_2p5,row.ci_upper_97p5],[i,i],color=col,lw=1.1);c.plot(row.median_full_advantage,i,'o',color=col,ms=4)
    c.set_yticks(range(6),labels,fontsize=7);c.invert_yaxis();c.axvline(0,c=GRAY,lw=.7,ls='--');c.set_xlabel('Full − alternative NSE');c.set_xlim(-.3,3.4);c.grid(axis='x',color=LIGHT,lw=.4);c.grid(axis='y',visible=False);c.text(.98,.015,'268 paired stations',transform=c.transAxes,ha='right',fontsize=7)
    d=f.add_subplot(gs[1,1]);panel(d,'d','Effect of calibration sample count')
    z=D['sample_size_learning_curve_summary'].query("cohort == 'common_max_size'").set_index('sample_size_label').loc[['5','10','20','all_available']]
    d.fill_between(range(4),z.station_q25_NSE,z.station_q75_NSE,color=BLUE,alpha=.17,lw=0);d.plot(range(4),z.station_median_NSE,'o-',color=BLUE,ms=4,lw=1);d.axhline(0,c=GRAY,lw=.7);d.set_xticks(range(4),['5','10','20','All']);d.set_xlabel('Calibration observations');d.set_ylabel('Station NSE');d.text(.02,.04,'55 valid stations\nSame holdout; band: station IQR\nAll: median 42.5 samples',transform=d.transAxes,fontsize=7)
    finish(f,'F02_sparse_constraints','Observation support and the contribution of sparse calibration','(a) Retained observations at 542 reconstruction stations, with hydrological boundaries for geographic context. (b) Eligibility counts; the horizontal axis is logarithmic. (c) Six principal comparisons on exact common held-out observations within each WSE assignment. Station-repeat NSE differences are reduced within station; points show across-station medians and bars show 95% intervals from 2,000 whole-station bootstrap resamples. Positive values favor the full model. The empirical baseline is highlighted. The complete ten-comparison summary is in Table S1. (d) Nested WSE-balanced calibration sizes on the same first-assignment holdout; 56 common stations, with valid NSE at 55. The band is the across-station interquartile range.','Existing support, paired-ablation and learning-curve tables',old='F02; F04a; F07c')


def validation():
    f=plt.figure(figsize=(7.1,5.5));gs=f.add_gridspec(2,2,left=.095,right=.975,bottom=.1,top=.93,hspace=.6,wspace=.40,height_ratios=[1.15,1])
    a=f.add_subplot(gs[0,0]);panel(a,'a','Held-out NSE within the WSE range');old.base_map(a)
    z=D['inside_station'].query("validation_scheme == 'wse_stratified'");g=old.xy(z)
    # Under-range map colors retain extreme negative NSE in a single open-ended class.
    sc=a.scatter(g.geometry.x,g.geometry.y,c=g.NSE.clip(-1,1),s=15,cmap='RdYlBu',vmin=-1,vmax=1,edgecolor='white',lw=.2)
    cb=f.colorbar(sc,ax=a,orientation='horizontal',fraction=.06,pad=.04);cb.set_ticks([-1,0,1],labels=['≤−1','0','1']);cb.set_label('Station NSE (209 eligible stations)',fontsize=7)
    sub=gs[0,1].subgridspec(2,1,height_ratios=[4.1,1],hspace=.06);b=f.add_subplot(sub[0]);lo=f.add_subplot(sub[1],sharex=b)
    p=D['predictions_domain'];p=p.query("validation_scheme == 'wse_stratified' and side == 'Inside'")
    p=p.groupby(['station_id','obs_id','Q_observed'],as_index=False).Q_calculated.median();cut=.01
    for ax,sel in [(b,p.Q_calculated.ge(cut)),(lo,p.Q_calculated.lt(cut))]:
        ax.scatter(p.loc[sel,'Q_observed'],p.loc[sel,'Q_calculated'],s=3.5,color=BLUE,alpha=.35,lw=0);ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.1,1e5);ax.grid(color=LIGHT,lw=.4)
    b.set_ylim(.01,2e5);lo.set_ylim(1e-17,.01);b.set_yticks([.01,1,100,10000]);lo.set_yticks([1e-16,1e-8]);b.plot([.1,1e5],[.1,1e5],'--',c=INK,lw=.7);b.tick_params(axis='x',labelbottom=False,bottom=False);b.spines['bottom'].set_visible(False);lo.spines['top'].set_visible(False);lo.set_xlabel('Observed discharge (m³/s)');b.set_ylabel('Estimated discharge (m³/s)');panel(b,'b','Unused WSE states inside the range');old.smallnote(b,f'{len(p):,} unique overpasses');lo.text(.99,.90,f'{int(p.Q_calculated.lt(cut).sum())} estimate <0.01',transform=lo.transAxes,ha='right',fontsize=6.5)
    for ax,y in [(b,0),(lo,1)]:
        for x in [0,1]:ax.plot([x-.014,x+.014],[y-.025,y+.025],transform=ax.transAxes,c=GRAY,lw=.7,clip_on=False)
    c=f.add_subplot(gs[1,0]);panel(c,'c','Positive pooled skill and station failures')
    series=[(D['inside_station'].query("validation_scheme == 'wse_stratified'").NSE,BLUE,'WSE inside'),(D['station_metrics'].query("validation_scheme == 'wse_stratified'").NSE,TEAL,'WSE all'),(D['station_metrics'].query("validation_scheme == 'temporal_forward'").NSE,ORANGE,'Temporal all')]
    for v,col,lab in series:
        v=v.dropna().sort_values().to_numpy();c.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=col,lw=1.15,label=f'{lab} (n={len(v)})')
    c.set_xscale('symlog',linthresh=1,linscale=.6);c.set_xlim(-1e7,1);c.set_xticks([-1e6,-1e3,-1,0,1],['−10⁶','−10³','−1','0','1']);c.set_ylim(0,1.03);c.axvline(0,c=GRAY,lw=.7,ls=':');c.set_xlabel('NSE (symmetric-log axis)');c.set_ylabel('Cumulative station fraction');c.legend(frameon=False,fontsize=6.5,loc='upper left')
    d=f.add_subplot(gs[1,1]);panel(d,'d','Sensitivity to the held-out assignment')
    z=D['domain_summary'].query("scheme == 'wse_stratified' and side == 'Inside'").sort_values('repeat');d.plot(z['repeat']+1,z.NSE,'o-',c=BLUE,lw=1,ms=3,label='Inside-range pooled NSE')
    z=D['experiment_overall_summary'].query("validation_scheme == 'wse_stratified'").sort_values('repeat_id');d.plot(z.repeat_id+1,z.station_median_NSE,'s--',c=TEAL,lw=1,ms=3,label='All-state station median');d.set_ylim(-1,1);d.set_yticks([-1,-.5,0,.5,1]);d.axhline(0,c=GRAY,lw=.7);d.set_xticks(range(1,11));d.set_xlabel('WSE assignment');d.set_ylabel('NSE');d.legend(frameon=False,fontsize=6.5,loc='lower right')
    old.AUDIT['F03_inside_scatter']={'observations':len(p),'below_floor_strip':int(p.Q_calculated.lt(cut).sum()),'NSE':1-((p.Q_observed-p.Q_calculated)**2).sum()/((p.Q_observed-p.Q_observed.mean())**2).sum()}
    finish(f,'F03_unused_states','Agreement on unused observations within the calibration WSE range','(a) Station-balanced NSE for within-range WSE tests; at least five pairs and nonzero observed and estimated variation are required within an assignment, then metrics are reduced within station. Colors below −1 are grouped in the lowest class. (b) Repeated predictions are median-collapsed only over assignments in which that observation was inside the calibration range. The explicit lower strip retains estimates below 0.01 m³/s. (c) Station NSE distributions for within-range WSE tests, all WSE tests, and all temporal tests. Cohorts differ and these curves are descriptive. (d) Per-assignment within-range pooled NSE and all-state station median NSE; the different aggregation levels are shown explicitly.','Saved predictions and validation metrics; newly aggregated within-range station metrics',old='F03 validation')


def domain():
    f=plt.figure(figsize=(7.1,5.7));gs=f.add_gridspec(2,2,left=.10,right=.97,bottom=.13,top=.85,hspace=.65,wspace=.38)
    for j,metric in enumerate(['NSE','MdAPE']):
        ax=f.add_subplot(gs[0,j]);panel(ax,chr(97+j),'Efficiency by WSE position' if metric=='NSE' else 'Relative errors by WSE position')
        for k,sc in enumerate(old.SC):
            z=D['domain_summary'].query('scheme == @sc').groupby('side')[metric].median().reindex(['Below','Inside','Above']);ax.plot(np.arange(3)+(k-.5)*.06,z,marker=old.MARK[k],color=old.COL[k],lw=1,ms=4)
        ax.set_xticks(range(3),['Below','Inside','Above']);ax.set_ylabel('Pooled NSE' if metric=='NSE' else 'Overpass MdAPE (%)')
        if metric=='NSE':ax.set_yscale('symlog',linthresh=1,linscale=.7);ax.set_yticks([-10,-1,0,1],['−10','−1','0','1']);ax.axhline(0,c=GRAY,lw=.7)
    for j,side in enumerate(['Above','Below']):
        ax=f.add_subplot(gs[1,j]);panel(ax,chr(99+j),'High-WSE error versus distance' if side=='Above' else 'Low-WSE error versus distance')
        for k,sc in enumerate(old.SC):
            z=D['distance_bins'].query('scheme == @sc and side == @side').set_index('bin').loc[['0–0.25','0.25–0.5','0.5–1','>1']];x=np.arange(4)+(k-.5)*.11
            ax.plot(x,z.factor,marker=old.MARK[k],c=old.COL[k],lw=1,ms=4)
            for xx,(_,r) in zip(x,z.iterrows()):
                ax.plot([xx,xx],[r.factor_ci_low,min(40,r.factor_ci_high)],c=old.COL[k],lw=.85)
                if r.factor_ci_high>40:ax.annotate('',(xx,39),(xx,30),arrowprops={'arrowstyle':'-|>','lw':.8,'color':old.COL[k]})
                ax.annotate(str(int(r.stations)),(xx,r.factor),xytext=(0,-13 if k==0 else 7),textcoords='offset points',ha='center',fontsize=6.5,color=old.COL[k])
        ax.set_yscale('log');ax.set_ylim(1,40);ax.set_yticks([1,2,5,10,20,40],['1','2','5','10','20','40']);ax.set_xticks(range(4),['(0, .25]','(.25, .5]','(.5, 1]','>1']);ax.set_xlabel('Boundary distance / calibration WSE span');ax.set_ylabel('Station median factor error');ax.axhline(2,c=GRAY,lw=.7,ls=':')
        if side=='Below':ax.text(.02,.95,'Arrow: upper CI >40\nWSE >1 bin: 9 stations',transform=ax.transAxes,ha='left',va='top',fontsize=6.3)
    old.legend(f,y=.99)
    finish(f,'F04_wse_domain','Discharge errors beyond the calibration WSE boundaries','(a–b) Separate below-range, inside-range and above-range tests. WSE values are medians of ten per-assignment pooled summaries; temporal values pool the later test blocks. High pooled NSE below the range coexists with large relative errors. (c–d) Boundary distance divided by the positive calibration WSE span, grouped at 0.25, 0.5 and 1. Factor error is max(Qestimated/Qobserved, Qobserved/Qestimated). Log errors are median-reduced within station-assignment and then within station before the across-station summary. Bars are 95% intervals from 2,000 whole-station bootstrap resamples; labels give stations in each bin. No model is refitted. An upward arrow shows a confidence limit beyond the plotted scale: for the nine-station WSE below-range >1 bin, the upper limit is 9.53 × 10¹⁶. Stations and observations can contribute to different bins across assignments; these comparisons do not isolate distance from station or event composition.','validation_predictions.csv; domain_summary.csv; distance_bins.csv',old='F07a–b; new boundary-distance aggregation')


support(); validation(); domain(); old.sos_fig(); old.width_fig(); old.parameter_fig()
# v15 numbering only: extreme diagnostics are Figure 5, SoS is Figure 6.
for ext in ['png','svg']:
    (OUT/'main'/f'F05_official_comparison.{ext}').rename(OUT/'main'/f'F06_official_comparison.{ext}')
    (OUT/'main'/f'F06_parameters.{ext}').rename(OUT/'supplement'/f'S02_effective_parameters.{ext}')
(OUT/'pdf/F05_official_comparison.pdf').rename(OUT/'pdf/F06_official_comparison.pdf')
(OUT/'pdf/F06_parameters.pdf').rename(OUT/'pdf/S02_effective_parameters.pdf')
for r in old.REG:
    if r['id']=='F05_official_comparison': r['id']='F06_official_comparison'
    if r['id']=='F06_parameters': r.update(id='S02_effective_parameters',group='supplement')
(OUT/'source/figure_manifest.json').write_text(json.dumps(old.REG,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'source/figure_audit.json').write_text(json.dumps(old.AUDIT,ensure_ascii=False,indent=2),encoding='utf-8')
print('Built v15 Figures 2, 3, 4, 6, S1, S2; Figures 1 and 5 have separate entry points')
