"""Pair methods on exactly the same retained held-out observations."""
import numpy as np
import pandas as pd


def paired_comparisons(predictions, cfg, pipeline):
    raw_parts=[];summary_parts=[];metric_rows=[];coverage=[]
    if predictions.empty:
        return (pd.DataFrame(),)*4
    data=predictions[(predictions.validation_scheme=='wse_stratified') & (predictions.repeat_id<cfg.ablation_repeat_count)]
    keys=['station_id','repeat_id','obs_id']
    full=data[data.method=='full']
    for comparator in pipeline.ABLATION_METHODS:
        other=data[data.method==comparator]
        shared=full[keys+['Q_observed','Q_calculated']].merge(
            other[keys+['Q_observed','Q_calculated']],on=keys,suffixes=('_full','_comparator'),validate='one_to_one')
        if shared.empty:continue
        if not np.allclose(shared.Q_observed_full,shared.Q_observed_comparator,rtol=0,atol=0):
            raise ValueError('Inconsistent observed discharge on paired support')
        rows=[]
        full_counts=full.groupby(keys[:2]).size();other_counts=other.groupby(keys[:2]).size()
        for (sid,repeat),group in shared.groupby(keys[:2]):
            coverage.append({'station_id':sid,'repeat_id':repeat,'comparator':comparator,
                             'full_native_count':int(full_counts.loc[(sid,repeat)]),
                             'comparator_native_count':int(other_counts.loc[(sid,repeat)]),
                             'common_count':len(group)})
            for method,side in [('full','full'),(comparator,'comparator')]:
                metrics=pipeline.core.calculate_metrics(group.Q_observed_full,group['Q_calculated_'+side])
                row={'station_id':sid,'repeat_id':repeat,'validation_scheme':'wse_stratified',
                     'method':method,'common_count':len(group),**metrics}
                rows.append(row);metric_rows.append({**row,'comparator':comparator})
        raw,summary=pipeline.paired_ablation_tests(pd.DataFrame(rows),cfg)
        if not raw.empty:
            raw['support']='exact_common_validation_observations';raw_parts.append(raw)
        if not summary.empty:
            summary['support']='exact_common_validation_observations';summary_parts.append(summary)
    return (pd.concat(raw_parts,ignore_index=True) if raw_parts else pd.DataFrame(),
            pd.concat(summary_parts,ignore_index=True) if summary_parts else pd.DataFrame(),
            pd.DataFrame(metric_rows),pd.DataFrame(coverage))
