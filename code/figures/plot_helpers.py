"""Rebuild the standalone SWOT figure set from saved, audited results.

No model fitting or changes to manuscript files. See code/README.md for commands.
"""
from pathlib import Path
import argparse, json, io, base64, textwrap, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter, ScalarFormatter, SymmetricalLogLocator
from matplotlib.transforms import Bbox
import geopandas as gpd

P=argparse.ArgumentParser();P.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]);P.add_argument('--cache',type=Path,default=Path(__file__).parent/'cache/data.pkl');P.add_argument('--pdf-work',type=Path,default=Path(__file__).parent/'cache/pdf');P.add_argument('--only',default='all');P.add_argument('--boundary',type=Path,default=Path('inputs/Class2_hydrounits.shp'))
ARGS=P.parse_args();OUT=ARGS.out;PDFWORK=ARGS.pdf_work;PDFWORK.mkdir(parents=True,exist_ok=True)
D=pd.read_pickle(ARGS.cache)
BLUE='#287C9E';ORANGE='#CE803A';TEAL='#25887F';GRAY='#83909B';INK='#243746';LIGHT='#DCE4E8';PURPLE='#7865A7'
SC=['wse_stratified','temporal_forward'];SL=['WSE-stratified','Temporal-forward'];COL=[BLUE,ORANGE];MARK=['o','s']
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.titleweight':'normal','axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#87939B','axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':3,'ytick.major.size':3,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'swot-20260927','savefig.facecolor':'white','figure.facecolor':'white','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
REG=[];AUDIT={};PANELS={}
def fig(w=7.1,h=5.4):return plt.figure(figsize=(w,h))
def panel(ax,letter,title):
    width=ax.get_position().width*ax.figure.get_figwidth()
    title=textwrap.fill(title,width=max(19,int(width*14)))
    ax.set_title(title,loc='left',pad=9)
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-13,10),textcoords='offset points',fontsize=10,fontweight='bold',va='bottom',ha='left',annotation_clip=False)
    ax.grid(axis='y',color=LIGHT,lw=.45,zorder=0)
    ax.set_axisbelow(True)
    return ax
def legend(fig,extra=False,y=.985):
    hs=[Line2D([],[],color=c,marker=m,lw=1,ms=3.5,label=l) for c,m,l in zip(COL,MARK,SL)]
    if extra:hs.insert(0,Line2D([],[],color=GRAY,lw=1,label='Reconstruction'))
    fig.legend(handles=hs,loc='upper center',bbox_to_anchor=(.5,y),ncol=len(hs),frameon=False)
def symax(ax,axis='y',th=1):
    getattr(ax,'set_'+axis+'scale')('symlog',linthresh=th,linscale=.65)
    target=getattr(ax,axis+'axis');tr=target.get_transform()
    extent=getattr(ax.dataLim,'interval'+axis)
    if np.isfinite(extent).all() and extent[1]>extent[0]:
        limits=tr.transform(extent);pad=(limits[1]-limits[0])*.08
        getattr(ax,'set_'+axis+'lim')(tr.inverted().transform([limits[0]-pad,limits[1]+pad]))
    locator=SymmetricalLogLocator(base=10,linthresh=th);locator.set_params(numticks=5);target.set_major_locator(locator)
def interval(ax,x,row,c,marker='o',horizontal=False,size=4):
    val=float(row['median']);lo=float(row.get('ci_low',np.nan));hi=float(row.get('ci_high',np.nan));n=int(row.get('stations',5))
    if not np.isfinite(val):return
    if np.isfinite(lo) and np.isfinite(hi):
        if not lo-1e-10<=val<=hi+1e-10:raise ValueError('Estimate outside interval')
        if horizontal:ax.plot([lo,hi],[x,x],color=c,lw=1.05)
        else:ax.plot([x,x],[lo,hi],color=c,lw=1.05)
    ax.plot(val if horizontal else x,x if horizontal else val,marker,ms=size,color=c,mfc=c if n>=5 else 'white',mew=.85,clip_on=True)
def smallnote(ax,text,xy=(.03,.96),ha='left'):
    ax.text(*xy,text,transform=ax.transAxes,ha=ha,va='top',fontsize=7,color=INK,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
def finish(f,key,title,caption,sources,group='main',old=''):
    f.canvas.draw();renderer=f.canvas.get_renderer()
    # Capture every chart panel separately for repositioning in draw.io.
    parts=[]
    objects=[*f.axes,*f.legends,*f.texts]
    visibility=[o.get_visible() for o in objects]
    boxes=[o.get_tightbbox(renderer) if v else None for o,v in zip(objects,visibility)]
    for i,(obj,box) in enumerate(zip(objects,boxes)):
        if box is None:continue
        box=box.transformed(f.dpi_scale_trans.inverted()).expanded(1.02,1.025)
        for other in objects:other.set_visible(other is obj)
        buf=io.StringIO();f.savefig(buf,format='svg',bbox_inches=box,facecolor='white')
        raw=buf.getvalue()
        parts.append({'svg':raw,'kind':'panel' if obj in f.axes else 'legend_or_note','x':float(box.x0*100),'y':float((f.get_figheight()-box.y1)*100),'w':float(box.width*100),'h':float(box.height*100)})
    for obj,v in zip(objects,visibility):obj.set_visible(v)
    PANELS[key]={'width':f.get_figwidth()*100,'height':f.get_figheight()*100,'panels':parts}
    dest=OUT/group;dest.mkdir(exist_ok=True)
    f.savefig(dest/f'{key}.svg',format='svg')
    f.savefig(dest/f'{key}.png',dpi=400)
    f.savefig(PDFWORK/f'{key}.pdf')
    REG.append({'id':key,'title':title,'caption':caption,'sources':sources,'group':group,'old_figures':old,'width_mm':round(f.get_figwidth()*25.4,1),'height_mm':round(f.get_figheight()*25.4,1)})
    plt.close(f);print('SAVED',key,flush=True)

META=D['station_metadata'].copy()
PROJ='+proj=lcc +lat_1=25 +lat_2=47 +lat_0=35 +lon_0=105 +datum=WGS84 +units=m'
GEO=gpd.GeoDataFrame(META,geometry=gpd.points_from_xy(META.longitude,META.latitude),crs='EPSG:4326').to_crs(PROJ)
BOUND=gpd.read_file(ARGS.boundary).to_crs(PROJ)
# Display-only 200 m simplification; station coordinates and analyses are unchanged.
BOUND.geometry=BOUND.geometry.simplify(200,preserve_topology=True)
EXT=[GEO.geometry.x.min()-170000,GEO.geometry.x.max()+170000,GEO.geometry.y.min()-170000,GEO.geometry.y.max()+170000]
def base_map(ax,background_ids=None):
    BOUND.plot(ax=ax,facecolor='#F8FAFA',edgecolor='#D5DDE0',linewidth=.23,zorder=0)
    g=GEO if background_ids is None else GEO[GEO.station_id.isin(background_ids)]
    ax.scatter(g.geometry.x,g.geometry.y,s=4.5,c='#D3D9DC',lw=0,zorder=1)
    ax.set(xlim=EXT[:2],ylim=EXT[2:]);ax.set_aspect('equal');ax.set_axis_off()
def xy(d):return GEO[['station_id','geometry']].merge(d,on='station_id',validate='one_to_one')
def sos_fig():
    f=fig(h=6.0);gs=f.add_gridspec(2,2,left=.11,right=.97,bottom=.10,top=.92,height_ratios=[1.2,1],wspace=.36,hspace=.55)
    a=f.add_subplot(gs[0,0]);panel(a,'a','Matched station coverage');base_map(a)
    p=D['sos_paired_station_metrics'];p=p[p.cohort.eq('high_confidence')&p.support.eq('pairwise')&p.rule.eq('baseline')&p.metric.eq('MdAPE')]
    counts={}
    for j,sc in enumerate(SC):
        ids=p.loc[p.validation_scheme.eq(sc),'station_id'].unique();g=GEO[GEO.station_id.isin(ids)];counts[sc]=len(g)
        a.scatter(g.geometry.x,g.geometry.y,s=15 if j==0 else 24,marker=MARK[j],facecolor=COL[j] if j==0 else 'none',edgecolor=COL[j],lw=.7,zorder=3+j,label=f'{SL[j]} (n={len(g)})')
    a.legend(frameon=False,fontsize=6.5,loc='lower left',bbox_to_anchor=(-.02,-.07))
    s=D['sos_paired_summary'];s=s[s.cohort.eq('high_confidence')&s.support.eq('pairwise')&s.rule.eq('baseline')]
    b=f.add_subplot(gs[0,1]);panel(b,'b','Product-specific MdAPE support')
    for j,sc in enumerate(SC):
        z=s[s.validation_scheme.eq(sc)&s.metric.eq('MdAPE')].set_index('product').loc[PRODUCTS]
        yy=np.arange(4)+(j-.5)*.30;b.barh(yy,z.stations,height=.27,color=COL[j])
        for y,n in zip(yy,z.stations):b.text(n+1.5,y,str(int(n)),va='center',fontsize=7)
    b.set_yticks(range(4),PRODUCTS);b.invert_yaxis();b.set_xlim(0,s.stations.max()*1.25);b.set_xlabel('Paired stations');b.grid(False)
    for j,metric in enumerate(['MdAPE','NSE']):
        ax=f.add_subplot(gs[1,j]);panel(ax,chr(99+j),'Local-model '+metric+' advantage')
        for k,sc in enumerate(SC):
            for x,prod in enumerate(PRODUCTS):
                z=s[s.validation_scheme.eq(sc)&s.metric.eq(metric)&s['product'].eq(prod)].iloc[0];interval(ax,x+(k-.5)*.17,z,COL[k],MARK[k])
        ax.set_xticks(range(4),PRODUCTS,rotation=25,ha='right');ax.axhline(0,color=GRAY,ls='--',lw=.8);symax(ax);ax.set_ylabel('Official − local MdAPE (pp)' if metric=='MdAPE' else 'Local − official NSE');ax.set_xlim(-.45,3.5)
    AUDIT['F05_map_union']=counts
    finish(f,'F05_official_comparison','Matched comparisons with official SoS discharge','(a) Union of stations eligible for a paired MdAPE comparison with at least one unconstrained SoS V3 product after the high-confidence crosswalk. Orange open squares identify temporal stations. (b) Product-specific station counts for MdAPE. (c–d) Median paired advantages of the local model over each product, with 95% station-bootstrap intervals. Each local–product comparison uses identical positive observations and at least four pairs per station; supports differ among products and may differ by metric. Positive values favor the local model. Both effect panels use symmetric-log axes. Local calibration uses gauge information; these comparisons do not establish general superiority in ungauged settings.','sos_paired_summary.csv; sos_paired_station_metrics.csv; station_metadata.csv',old='Figure 11; Table 3')


def parameter_fig():
    f=fig(h=5.7);axs=f.subplots(2,3);f.subplots_adjust(left=.09,right=.91,bottom=.14,top=.90,wspace=.62,hspace=.80)
    p=D['parameters'];m=D['parameter_multistart_summary']
    for ax,col,letter,title,bins in [(axs[0,0],'n','a','Effective resistance n',np.linspace(.01,.15,20)),(axs[0,1],'beta','b','Area-increment multiplier β',np.geomspace(.25,4,20))]:
        panel(ax,letter,title);ax.hist(p[col],bins=bins,color=BLUE,edgecolor='white',lw=.5);ax.axvline(p[col].median(),color=ORANGE,ls='--',lw=1);ax.set_ylabel('Stations');ax.set_xlabel(col if col=='n' else 'β');smallnote(ax,f'Median = {p[col].median():.3f}')
        if col=='beta':ax.set_xscale('log');ax.set_xticks([.25,.5,1,2,4],['0.25','0.5','1','2','4'])
    ax=axs[0,2];panel(ax,'c','Joint parameter estimates');sc=ax.scatter(p.n,p.beta,c=p.objective_loss,cmap='viridis',s=8,alpha=.7,lw=0);ax.set_xlabel('n');ax.set_ylabel('β');ax.set_yscale('log');ax.set_yticks([.25,.5,1,2,4],['0.25','0.5','1','2','4']);f.colorbar(sc,ax=ax,pad=.035,fraction=.07).set_label('Objective loss',fontsize=7)
    ax=axs[1,0];panel(ax,'d','Fits close to bounds');keys=['n','beta','depth','width0'];yl=['n','β','Depth$_0$','Width$_0$']
    low=[p[k+'_at_lower_bound'].astype(bool).mean()*100 for k in keys];high=[p[k+'_at_upper_bound'].astype(bool).mean()*100 for k in keys]
    ax.barh(np.arange(4)-.15,low,height=.28,color=TEAL,label='Lower');ax.barh(np.arange(4)+.15,high,height=.28,color=ORANGE,label='Upper');ax.set_yticks(range(4),yl);ax.set_xlabel('Fits (%)');ax.set_xlim(0,80);ax.legend(frameon=False,ncol=2,fontsize=6.5,loc='lower right');ax.invert_yaxis();ax.grid(False)
    ax=axs[1,1];panel(ax,'e','Across-start variability');cols=['objective_loss_cv','A0_cv','depth0_cv','width0_cv'];labels=['Loss','A$_0$','Depth$_0$','Width$_0$']
    vals=[m[c][np.isfinite(m[c])&(m[c]>0)].values for c in cols]
    ax.boxplot(vals,tick_labels=labels,showfliers=False,patch_artist=True,boxprops={'facecolor':'#D5E6EB','edgecolor':BLUE,'linewidth':.8},medianprops={'color':INK,'linewidth':1},whiskerprops={'color':BLUE,'linewidth':.8},capprops={'color':BLUE,'linewidth':.8});ax.set_yscale('log');ax.set_ylabel('Coefficient of variation');ax.tick_params(axis='x',rotation=25)
    ax=axs[1,2];panel(ax,'f','Sensitivity to resistance n');s=D['n_sensitivity_summary'];s=s[s.grid_type.eq('relative_multiplier_summary')].sort_values('n_multiplier_requested');ax.plot(s.n_multiplier_requested,s.station_median_NSE,'o-',color=BLUE,lw=1,ms=3);ax.axvline(1,color=GRAY,ls='--',lw=.8);ax.set_xlabel('Requested multiplier\nof fitted n');ax.set_ylabel('Station-median NSE');ax.set_xticks([.5,.8,1,1.2,1.5]);smallnote(ax,'At ×1.2: 65.1% clipped\nGeometry held fixed',(.98,.05),ha='right');ax.texts[-1].set_va('bottom')
    AUDIT['F06']={'n_median':float(p.n.median()),'n_upper_pct':high[0],'beta_median':float(p.beta.median()),'beta_lower_pct':low[1]}
    finish(f,'F06_parameters','Effective parameters and uncertainty','All panels concern full-record reconstruction at 542 stations. (a–c) Fitted n and β, including their joint distribution colored by objective loss. (d) Fractions within 1% of a transformed search interval from each bound. (e) Across-station distributions of coefficients of variation across three optimization starts; boxes show the interquartile range, whiskers extend to 1.5 IQR, and outliers are omitted from this box-plot panel only. Nonpositive CVs cannot appear on the log axis. (f) Reconstruction sensitivity to a requested multiplier of n, with geometry and other inputs fixed. Bound clipping makes the perturbations asymmetric: 65.1% of stations are clipped at the requested ×1.2. Numerical stability does not establish physical identifiability.','parameter_estimates.csv; parameter_multistart_summary.csv; n_sensitivity_summary.csv',old='Figure 8; sensitivity output')


def width_fig():
    f=fig(h=5.0);axs=f.subplots(2,2);f.subplots_adjust(left=.10,right=.98,bottom=.13,top=.88,wspace=.35,hspace=.55)
    hs=[Line2D([],[],color=GRAY,marker='o',ls='',ms=3,label='Calibration anchors'),Line2D([],[],color=ORANGE,ls='--',lw=1,label='Isotonic fit'),Line2D([],[],color=BLUE,lw=1.5,label='Growth-capped fit')];f.legend(handles=hs,loc='upper center',bbox_to_anchor=(.5,.98),ncol=3,frameon=False)
    for i,(ax,sid) in enumerate(zip(axs.flat,D['width_example_ids'])):
        z=D['width_curves'];z=z[z.station_id.eq(sid)].sort_values('curve_wse');panel(ax,chr(97+i),f'Station {sid}')
        x=z.curve_wse-z.curve_wse.min();ax.scatter(x,z.raw_width,s=10,color=GRAY,alpha=.75);ax.plot(x,z.isotonic_width,color=ORANGE,ls='--',lw=1);ax.plot(x,z.curve_width,color=BLUE,lw=1.4);ax.set_xlabel('WSE above lowest anchor (m)');ax.set_ylabel('Width (m)');n=int(z.growth_capped.astype(bool).sum());smallnote(ax,f'{n} capped anchors')
    finish(f,'S01_width_constraints','Examples of calibration width constraints','Four reconstruction stations with the largest counts of growth-capped anchors. Gray points are calibration anchors, dashed orange lines the isotonic fit after the initial-width check, and solid blue lines the growth-capped relation. WSE is offset to each station’s lowest anchor for readability. These deliberately selected affected cases illustrate the constraint and do not estimate average performance improvement.','width_curve_anchor_audit.csv; parameter_estimates.csv',group='supplement',old='Figure S1')


