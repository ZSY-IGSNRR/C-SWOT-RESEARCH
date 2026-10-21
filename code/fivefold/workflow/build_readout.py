"""Build the concise Chinese result note from verified aggregate CSVs."""
from pathlib import Path
import html,json,sys
from _paths import work_root
ROOT=work_root()
import numpy as np
import pandas as pd
T=ROOT/'tables';NEW='wse_stratified_5fold';OLD='wse_stratified_10_repeats'
def read(name):return pd.read_csv(T/(name+'.csv'))
def score(row,metric):
    precision=3 if metric=='NSE' else 0
    return f'{row[metric+"_median"]:,.{precision}f} ± {row[metric+"_half_IQR"]:,.{precision}f}'
def only(frame,**filters):
    for k,v in filters.items():frame=frame.loc[frame[k].eq(v)]
    assert len(frame)==1,(filters,len(frame))
    return frame.iloc[0]
md=[];ht=[]
def heading(text,level=2):
    md.extend(['#'*level+' '+text,'']);ht.append(f'<h{level}>{html.escape(text)}</h{level}>')
def paragraph(text):
    md.extend([text,'']);ht.append('<p>'+html.escape(text)+'</p>')
def table(columns,rows):
    md.extend(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |'])
    md.extend('| '+' | '.join(str(v) for v in row)+' |' for row in rows);md.append('')
    ht.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>')
def link(label,path):
    md.extend([f'[{label}]({path})','']);ht.append(f'<p><a href="{html.escape(path)}">{html.escape(label)}</a></p>')

s=read('cross_gauge_summary');f=read('cross_gauge_metrics_by_split');u=read('unique_observation_and_stacked_metrics')
ss=read('station_unique_observation_summary');trim=read('extreme_residual_summary')
qa=json.loads((ROOT/'checks/final_independent_audit.json').read_text(encoding='utf-8'))
heading('V15 标准五折 WSE 分层交叉验证结果',1)
paragraph('已完成422个站点的标准五折CV。每次约80%率定、20%验证，7,615条符合条件的配对观测各验证一次。完整模型、原有消融和两个固定参数对照共13种设置，得到27,430项站点—折—方法记录，拟合失败数为0。原V15数据、QC、宽度改正、水力模型及优化设置均保持原样。论文和回应已按五折结果同步更新，原文件保留。')
heading('1. 完整模型：与原十次划分比较')
rows=[]
domains={'all':'全部验证观测','within':'训练WSE范围内','above':'高于训练WSE范围','below':'低于训练WSE范围'}
for domain,label in domains.items():
    old=only(s,scheme=OLD,support='native',domain=domain,method='full')
    new=only(s,scheme=NEW,support='native',domain=domain,method='full')
    rows.append([label,score(old,'NSE'),score(new,'NSE'),score(old,'RMSE'),score(new,'RMSE')])
table(['分析范围','原十次NSE','新五折NSE','原十次RMSE','新五折RMSE'],rows)
paragraph('这里先在每一折内拼接各站验证观测，计算一个跨站NSE和RMSE，再报告五折的中位数 ± IQR/2。RMSE单位均为m³/s。IQR/2描述折间差异，不是置信区间。全部观测的NSE中位数略升，范围内NSE中位数下降，范围内RMSE也下降；各折实测Q的方差及观测组成不同，因此两个指标可以朝不同方向变化。')
paragraph('原十次结果来自70%/30%重复随机划分，共433站、22,562次验证预测，涉及7,069条唯一观测。新旧站点重合404站，新五折新增18站、移出29站。五折改变了训练比例、站点资格和观测覆盖，不能把成绩差异全部解释为折数的作用。共同站点与共同观测的敏感性比较另存CSV。')
link('共同站点/观测比较表','tables/old_new_intersection_summary.csv')

heading('2. 必须同时保留逐折和合并五折成绩')
base=f.loc[f.scheme.eq(NEW)&f.support.eq('native')&f.method.eq('full')&f.domain.eq('all')].sort_values('fold_id')
table(['验证折','观测数','NSE','RMSE (m³/s)'],[[int(r.fold_id)+1,int(r.nobs),f'{r.NSE:.3f}',f'{r.RMSE:,.0f}'] for _,r in base.iterrows()])
rows=[]
for domain,label in domains.items():
    r=only(u,scheme=NEW,support='native',domain=domain,method='full',aggregation='single_out_of_fold_prediction')
    rows.append([label,int(r.nobs),f'{r.NSE:.3f}',f'{r.RMSE:,.0f}'])
table(['合并五折，每条观测仅计算一次','观测数','NSE','RMSE (m³/s)'],rows)
paragraph('第4折存在很大的平方误差，使全观测NSE降至−5.654。五折中位数及IQR均不能充分显示这一折的失败；合并五折后的全观测NSE为−0.706。因此，五折结果应同时报告逐折成绩和完整OOF成绩，不能只依据0.466 ± 0.090判断模型稳定。高于训练范围的合并NSE为−5.999，表明高水位外推仍然存在严重失败。')
st=only(ss,scheme=NEW,support='native',domain='all',method='full')
paragraph(f'逐站把五折的独立验证预测合并后，418个可评站的NSE中位数为{score(st,"NSE")}，NSE>0的站点占{100*st.positive_NSE_fraction:.1f}%。另外4站的实测Q没有变化，NSE分母为零。本次完整模型没有恒定预测，允许恒定预测的数学NSE定义与原资格规则得到相同结果。')

heading('3. 宽度处理与水力参数率定的收益')
labels={'observed_fixed':'观测河宽，四参数固定','corrected_fixed':'改正河宽，四参数固定','no_monotonic':'观测河宽，四参数率定','n_only_fixed_geometry':'仅率定n，几何固定','fixed_reference_n':'固定参考n','beta_fixed_1':'固定β=1','full':'完整联合率定','empirical_rating_curve':'经验WSE–Q曲线'}
rows=[]
for method,label in labels.items():
    r=only(s,scheme=NEW,support='all_methods_common',domain='all',method=method)
    rows.append([label,score(r,'NSE'),score(r,'RMSE')])
table(['处理方式，相同验证观测','五折NSE','五折RMSE (m³/s)'],rows)
paragraph('宽度改正与联合率定带来的收益仍然清楚。两个四参数固定对照沿用事先约定的V15参考n及预设几何规则，它们是本研究的固定配置对照，不代表Durand等方法或官方产品的普遍性能。经验WSE–Q曲线在本地有流量观测的交叉验证中仍是强基准，本次也优于完整水力模型。')

heading('4. 与四种官方产品的相同观测比较')
p=read('products_fold_summary');po=read('products_OOF_metrics');rows=[]
for model in ['Local','Consensus','MetroMan','MOMMA','SIC4DVar']:
    a=only(p,support='common_four',domain='all',model=model,metric='NSE')
    b=only(p,support='common_four',domain='all',model=model,metric='RMSE')
    c=only(po,support='common_four',domain='all',model=model)
    rows.append([model,f'{a["median"]:.3f} ± {a.half_IQR:.3f}',f'{b["median"]:,.0f} ± {b.half_IQR:,.0f}',f'{c.NSE:.3f}'])
table(['方法','五折NSE','五折RMSE (m³/s)','合并五折NSE'],rows)
paragraph('共同样本为108站、794条观测，各方法逐键使用同一份实测Q。本方法在该共同样本上仍优于四种官方产品；本地模型使用了率定流量，而产品为此处采用的无约束估计，这一差异需要在解释时保留。共同样本覆盖研究网络的一部分，不能把表中成绩直接推广到全部站点。官方产品更广可用样本的独立统计未受CV改变影响，已保留其原汇总与来源。')

heading('5. 极端误差、外推距离及参数边界')
r=only(trim,scheme=NEW,support='native',domain='all',method='full')
paragraph(f'每折最大的约1%绝对残差贡献了平方误差总量的中位{100*r.removed_SSE_fraction_median:.1f}%。事后排除这些记录、不重新率定时，五折NSE为{score(r,"NSE")}，RMSE为{score(r,"RMSE")} m³/s。这只是已知误差的影响诊断，需要实测Q，不能作为实际应用中自动剔除记录的规则。')
paragraph('上界外四个距离箱的等站点factor error中位数依次为1.892、2.592、3.373和3.272。误差总体较近边界时更高，最后一箱略有回落，不能继续写为严格单调增加。最远下界外仅6站，分布很宽，不能据此确定通用外推阈值。')
paragraph('2,110项完整模型的站点—折参数中，n接近上界的比例为62.61%，β接近下界为33.93%，任一参数接近边界为69.10%。这些标记沿用V15在优化坐标跨度内1%的定义。近界描述参数约束状态，不等于优化失败。')
link('外推距离统计','tables/extrapolation_factor_error_OOF_summary.csv')
link('参数边界统计','tables/parameter_boundary_summary.csv')

heading('6. 图像及可复核文件')
for file,title in [('fivefold_validation_overview','五折完整模型'),('fivefold_method_comparison','宽度与参数对照'),('fivefold_vs_ten_repeats','新旧验证设计，共同站点')]:
    md.extend([f'![{title}](figures/{file}.png)','',f'[PDF](figures/{file}.pdf)',''])
    ht.append(f'<figure><a href="figures/{file}.pdf"><img src="figures/{file}.png" alt="{title}"></a><figcaption>{title}；点为各折/次成绩，线为Q25–Q75，短横线为中位数。</figcaption></figure>')
link('完整13种方法与各水位域统计','tables/readout.md')
link('独立完整性审计','checks/final_independent_audit.json')
link('实验设计','fivefold_design.json')
paragraph('所有逐观测实测Q、配对记录和模型检查点均保存在private/，该目录是本地受限材料。原论文、回复文件和原十次实验结果均未覆盖。')
(ROOT/'RESULTS_五折与原十次对比.md').write_text('\n'.join(md),encoding='utf-8')
css='body{margin:0 auto;max-width:1280px;padding:40px 28px;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#25313d;background:#fff}h1{font-size:30px;line-height:1.35}h2{font-size:22px;margin-top:36px;border-bottom:1px solid #dce4e9;padding-bottom:10px}p{max-width:1120px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #dce4e9;text-align:left;white-space:nowrap}th{background:#eef4f7}tbody tr:nth-child(even){background:#f8fafb}a{color:#17659c}figure{margin:30px 0}img{max-width:100%;height:auto}figcaption{font-size:14px;color:#52616c}footer{margin-top:40px;color:#52616c;font-size:13px}'
(ROOT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V15标准五折验证</title><style>'+css+'</style><body>'+''.join(ht)+'<footer>V15 · 2026-10-19 · 本地复核版，修订稿已完成</footer></body></html>',encoding='utf-8')
print(json.dumps({'result_note':str(ROOT/'RESULTS_五折与原十次对比.md'),'viewer':str(ROOT/'index.html'),'status':'PASS'},ensure_ascii=False))
