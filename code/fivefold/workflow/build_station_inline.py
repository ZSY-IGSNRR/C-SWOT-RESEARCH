"""Bounded, aggregate-only input for the native inline distribution chart."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from _paths import work_root
ROOT=work_root()
V=ROOT/'readouts'
V.mkdir(parents=True,exist_ok=True)
V.mkdir(exist_ok=True,parents=True)
d=pd.read_csv(ROOT/'tables/station_analysis_gauge_metrics.csv')
d=d.loc[d.method.eq('full')&d.domain.eq('all')&d.NSE.notna()]
bins=[-np.inf,-10,-1,-.5,0,.5,np.inf]
labels=['小于−10','−10至−1','−1至−0.5','−0.5至0','0至0.5','不低于0.5']
counts=pd.cut(d.NSE,bins=bins,labels=labels,right=False).value_counts(sort=False)
rows=[{'NSE区间':label,'站点数':int(counts[label]),'占比':float(counts[label]/len(d))} for label in labels]
source={'label':'V15标准五折：站点OOF统计','files':[{'label':'station_analysis_gauge_metrics.csv'}],
    'metricDefinitions':[{'name':'站点NSE分布','definition':'每站合并五折独立验证预测后计算一个NSE，再按不等宽区间计数；每站等权。'}],
    'filters':['方法：完整联合率定','样本：至少5对有效验证观测且实测与预测均有变化','范围：全部验证WSE状态'],
    'caveats':['各区间左闭右开，最后一箱包含0.5；区间宽度不同，柱高仅表示站点数。','全部极端负NSE均保留在最左区间，4个实测流量恒定站点无法计算NSE。']}
payload={'schemaVersion':1,'id':'gauge-nse-distribution-fivefold','title':'站点NSE分布（418站，区间不等宽）',
    'chart':{'type':'bar','x':'NSE区间','y':'站点数','showValues':True,'showXAxisLabel':False,'yLabel':'站点数','startAtZero':True},
    'rows':rows,'source':source,'height':330,'theme':'codex-classic'}
(V/'station-nse-distribution.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
receipt={'schemaVersion':1,'items':[{'id':'gauge-nse-distribution','title':'418个站点的NSE分布',
    'queries':[{'id':'fivefold-gauge-bins','source':source,'rows':rows,
        'columns':['NSE区间','站点数','占比'],
        'preview':{'kind':'aggregate','note':'完整418站按六个区间汇总，已包含全部极端分数。','totalRows':6}}]}]}
(V/'station-nse-distribution-sources.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rows,ensure_ascii=True))
