"""Station-level Chinese note and bounded source evidence for the chat."""
from pathlib import Path
import json
import pandas as pd
from _paths import work_root
ROOT=work_root();T=ROOT/'tables'
V=ROOT/'readouts'
V.mkdir(parents=True,exist_ok=True)
s=pd.read_csv(T/'station_analysis_summary.csv');p=pd.read_csv(T/'products_OOF_gauge_summary.csv')
c=pd.read_csv(T/'station_analysis_correlations.csv')
def fmt(row,key):return f'{row[key+"_median"]:.3f} ± {row[key+"_half_IQR"]:.3f}'
def get(method,domain='all'):return s.loc[s.method.eq(method)&s.domain.eq(domain)].iloc[0]
lines=['# 标准五折CV：站点级统计分析','',
'每站先合并五折互不重复的验证预测，计算一个NSE和RMSE，再在站点间等权汇总中位数 ± IQR/2。这里的IQR是站点分布的离散程度，区别于跨站计算后五折之间的IQR。NSE与主要RMSE表使用同一批可计算NSE的站点。','',
'|范围|可评价站数|站点NSE|站点RMSE (m³/s)|NSE>0|','|---|---:|---:|---:|---:|']
primary_rows=[]
for domain,label in [('all','全部验证观测'),('within','训练WSE范围内')]:
    r=get('full',domain);n=int(r.NSE_evaluable_stations)
    lines.append(f'|{label}|{n}|{fmt(r,"NSE")}|{fmt(r,"RMSE_same_NSE_gauges")}|{100*r.positive_NSE_fraction:.1f}%|')
    primary_rows.append({'范围':label,'站点数':n,'NSE中位数':r.NSE_median,'NSE_IQR半宽':r.NSE_half_IQR,
        'RMSE中位数_m3s':r.RMSE_same_NSE_gauges_median,'RMSE_IQR半宽_m3s':r.RMSE_same_NSE_gauges_half_IQR,
        'NSE正值比例':r.positive_NSE_fraction})
lines+=['','4站的实测流量恒定，全观测NSE不可计算；保留这些站时，全部422站的RMSE为37.958 ± 75.622 m³/s。高于/低于训练WSE范围的观测每站均不足5条，不能按现有规则计算站点NSE。',
'','站点RMSE较小与站点NSE为负并不矛盾。每站NSE = 1 − (RMSE / 实测Q总体标准差)²，负值表示平方误差超过该站实测流量的离均差平方和。很多站点流量变化幅度较小，绝对误差不大也可能得到负NSE。','',
'## 与消融和经验曲线比较','',
'|模型|站点NSE|同418站RMSE (m³/s)|NSE>0|','|---|---:|---:|---:|']
for method,label in [('full','完整联合率定'),('no_monotonic','无河宽改正'),('fixed_reference_n','固定参考n'),('beta_fixed_1','固定β=1'),('n_only_fixed_geometry','仅率定n、几何固定'),('empirical_rating_curve','经验WSE–Q曲线')]:
    r=get(method);lines.append(f'|{label}|{fmt(r,"NSE")}|{fmt(r,"RMSE_same_NSE_gauges")}|{100*r.positive_NSE_fraction:.1f}%|')
lines+=['','完整模型相较无河宽改正，在73.0%的共同站点上NSE更高；相较仅率定n的固定几何对照，也是73.0%。相较固定参考n和固定β，对应比例分别为60.8%和56.9%。相较经验WSE–Q曲线，完整模型在39.2%的站点更优，因此本次站点结果没有显示水力模型全面超过本地经验曲线。',
'','## 与四种官方产品比较','',
'共同产品样本共有108站794条观测；按每站至少5条及五个模型均可计算NSE的条件，站点级成绩表使用同一批67站679条观测。','',
'|方法|站点NSE|同67站RMSE (m³/s)|','|---|---:|---:|']
product_rows=[]
for model,label in [('Local','本文方法'),('Consensus','Consensus'),('MetroMan','MetroMan'),('MOMMA','MOMMA'),('SIC4DVar','SIC4DVar')]:
    r=p.loc[p.support.eq('common_four')&p.domain.eq('all')&p.model.eq(model)&p.metric.eq('NSE')].iloc[0]
    m=p.loc[p.support.eq('common_four')&p.domain.eq('all')&p.model.eq(model)&p.metric.eq('RMSE')].iloc[0]
    assert r.N==m.N==67
    lines.append(f'|{label}|{r["median"]:.3f} ± {r.half_IQR:.3f}|{m["median"]:.2f} ± {m.half_IQR:.2f}|')
    product_rows.append({'方法':label,'站点数':67,'NSE中位数':r['median'],'NSE_IQR半宽':r.half_IQR,'RMSE中位数_m3s':m['median'],'RMSE_IQR半宽_m3s':m.half_IQR})
lines+=['','本文方法分别在80.6%、85.1%、80.6%和73.1%的共同站点上优于Consensus、MetroMan、MOMMA和SIC4DVar。所有方法的站点NSE中位数仍为负，说明相对改善与逐站精度仍不足可以同时成立。本地模型率定使用实测Q，与这些无约束产品的地面信息条件不同。',
'','## 数据量及站间差异','',
'站点NSE与观测数量的Spearman相关系数ρ=0.086，1000次站点bootstrap的95%区间为−0.015至0.180。至少10条观测时，284个可评站NSE为−0.353 ± 0.799；至少20条时，81站为−0.299 ± 1.194，NSE>0仍只有约26%。因此，观测数量少不能单独解释大多数站点NSE偏低。',
'','与NSE关联较强的是参数近界频率，β为ρ=−0.538，n为−0.432，depth₀为−0.430；这些是关联，不能证明边界命中导致误差。原始width变异系数、宽度改正幅度对应ρ为−0.261和−0.206，width Hampel异常比例仅为0.026。统计不支持将全部站间差异归结于width异常。',
'','RMSE与实测Q均值、标准差的ρ分别为0.822和0.843，说明绝对RMSE明显受流量规模影响。NSE与Q标准差的相关还包含指标定义上的数学耦合，不能作为数据质量的独立因果证据。未计算相关性p值或作显著性宣称。',
'','本次站点OOF口径与旧稿“每折先算站点NSE、再取站点中位数”的口径不同，不宜将两者直接比较来宣称改进。论文与回复文件未修改。','',
'[逐站汇总表](tables/station_analysis_summary.csv) · [方法配对增益](tables/station_analysis_paired_gains.csv) · [相关性分析](tables/station_analysis_correlations.csv) · [独立审计](checks/station_analysis_audit.json)','']
(ROOT/'RESULTS_站点级统计.md').write_text('\n'.join(lines),encoding='utf-8')
def item(id,title,file,definition,rows,caveats):
    return {'id':id,'title':title,'queries':[{'id':id+'-table','source':{'label':title,
        'files':[{'label':file}],'metricDefinitions':[{'name':title,'definition':definition}],
        'caveats':caveats},'rows':rows,'columns':list(rows[0]),'preview':{'kind':'aggregate','note':'已核对的站点等权汇总，未包含逐观测实测流量。','totalRows':len(rows)}}]}
rows=c.loc[c.minimum_oof_pairs.eq(5)&((c.metric.eq('NSE')&c.feature.isin(['n_oof','beta_boundary_hit_fraction','n_boundary_hit_fraction']))|(c.metric.eq('RMSE')&c.feature.isin(['Q_mean','Q_sd_population'])))].copy()
driver_rows=rows[['metric','feature','n_gauges','Spearman_rho','bootstrap_95ci_low','bootstrap_95ci_high']].to_dict('records')
if not driver_rows:
    r=c.loc[c.minimum_oof_pairs.eq(5)&c.metric.eq('NSE')&c.feature.eq('n_oof')].iloc[0]
    driver_rows=[r[['metric','feature','n_gauges','Spearman_rho','bootstrap_95ci_low','bootstrap_95ci_high']].to_dict()]
receipt={'schemaVersion':1,'items':[
    item('gauge-primary','完整模型的站点级表现','station_analysis_summary.csv','每站合并五折独立验证预测后计算一次NSE和RMSE，站点之间等权报告中位数及IQR/2。',primary_rows,['NSE和RMSE使用相同可评价站点；全部观测418站、范围内417站。','范围外每站样本不足5条，不计算站点NSE。']),
    item('gauge-products','相同站点的官方产品比较','products_OOF_gauge_summary.csv','在共同67站679条观测上计算各站分数，再等权汇总站点分布。',product_rows,['本地模型使用率定流量，官方产品使用此处采用的无约束估计。','共同样本并不代表全部研究网络。']),
    item('gauge-drivers','样本数量和站点误差的关联','station_analysis_correlations.csv','以完整模型每站OOF分数和站点数据特征计算平均秩Spearman相关，区间为1000次站点bootstrap的95%区间。',driver_rows,['相关不能证明因果，NSE与Q方差存在数学耦合。','未计算p值，未作显著性宣称。'])]}
(V/'station-analysis-sources.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','note':str(ROOT/'RESULTS_站点级统计.md')},ensure_ascii=True))
