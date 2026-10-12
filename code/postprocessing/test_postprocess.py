"""Focused tests of metric arithmetic, observation grain and SoS matching."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from common import metrics, assign_groups, aggregate_units
from analyze_postprocess import make_flags, association_analysis
from sos_analysis import match_sos, PRODUCTS

class Tests(unittest.TestCase):
    def test_metrics_independent_arithmetic(self):
        r=metrics([1,2,3,4,5],[1,2,3,4,7])
        self.assertAlmostEqual(r['NSE'],.6)  # SSE=4, SST=10
        self.assertAlmostEqual(r['MAE'],.4)
        self.assertAlmostEqual(r['RMSE'],np.sqrt(.8))
        self.assertEqual(r['MdAPE'],0)
        self.assertTrue(np.isnan(metrics([1,2],[1,2])['MAE']))
        self.assertEqual(metrics([1,2],[0,2],min_pairs=1)['n'],2)

    def test_scale_boundaries_and_missing(self):
        d=pd.DataFrame(dict(gauge_Q50=[0,10,100,1000,1001],gauge_days=[337,337,337,269,337],
            sword_width_m=[99,100,200,500,0],longitude=[100]*5,latitude=[40]*5))
        g=assign_groups(d)
        self.assertEqual(g.flow_group.iloc[0],'Recorded zero')
        self.assertEqual(g.flow_group.iloc[1],'(0,10]')
        self.assertTrue(pd.isna(g.flow_group.iloc[3]))
        self.assertEqual(g.width_group.iloc[1],'100-<200')
        self.assertTrue(pd.isna(g.width_group.iloc[4]))

    def test_flags_repeat_grain_and_temporal_pooling(self):
        rows=[];params=[]
        for repeat in [0,1]:
            for i in range(6):
                rows.append(dict(station_id='1',validation_scheme='wse_stratified',obs_id=str(i),repeat_id=repeat,
                    fold_id=0,method='full',time=pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(days=i),
                    Q_observed=10+i,Q_calculated=(200 if i==0 else 10+i),slope_used=(20 if i==0 and repeat==0 else 1),
                    WSE=i,width_raw=20,width_used=20))
            params.append(dict(station_id='1',validation_scheme='wse_stratified',repeat_id=repeat,fold_id=0,
                train_slope_fallback=1,slope_positive_count=5,slope_used_global_fallback=False,method='full'))
        with tempfile.TemporaryDirectory() as tmp:
            full,obs,base=make_flags(pd.DataFrame(rows),pd.DataFrame(params),Path(tmp),5)
            self.assertEqual(len(obs),6)
            self.assertEqual(len(full),12)
            self.assertTrue(obs.loc[obs.obs_id.eq('0'),'slope10'].item()) # half of repeats
            self.assertTrue(obs.loc[obs.obs_id.eq('0'),'error10'].item())
            self.assertEqual(len(base),2)
        temporal=pd.DataFrame(rows[:6]);temporal['validation_scheme']='temporal_forward'
        temporal['fold_id']=[0,0,1,1,2,2]
        self.assertEqual(aggregate_units(temporal).n.item(),6)

    def test_sos_time_and_crosswalk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            pd.DataFrame([dict(station_id='1',mapping_status='remapped_high',mapping_confidence='high',
                v16_reach_id='99',v17b_reach_id='10',nearest_v17b_node_id='101')]).to_csv(root/'audit.csv',index=False)
            records=[dict(reach_id='99',sword_version=16,time_str='2024-01-01T00:00:10Z',
                **{x:12 for x in PRODUCTS})]
            # Conflicting future rows are irrelevant to this January prediction.
            records += [dict(reach_id='99',sword_version=16,time_str='2024-07-01T00:00:10Z',
                **{x:value for x in PRODUCTS}) for value in [20,30]]
            pd.DataFrame(records).to_csv(raw/'99.csv',index=False)
            obs=pd.DataFrame([dict(station_id='1',validation_scheme='wse_stratified',obs_id='one',
                time=pd.Timestamp('2024-01-01',tz='UTC'),Q_model=123,Q_observed=11,reach_id='10',node_id='101')])
            matched=match_sos(obs,SimpleNamespace(station_audit=root/'audit.csv',sos_raw_dir=raw),root)
            self.assertEqual(matched.Q_model.item(),123)
            self.assertEqual(matched.time_offset_seconds.item(),10)
            self.assertEqual(matched.sos_consensus_q.item(),12)
            records.append(dict(reach_id='99',sword_version=16,time_str='2024-01-01T00:00:10Z',
                **{x:99 for x in PRODUCTS}))
            pd.DataFrame(records).to_csv(raw/'99.csv',index=False)
            with self.assertRaisesRegex(ValueError,'Conflicting SoS'):
                match_sos(obs,SimpleNamespace(station_audit=root/'audit.csv',sos_raw_dir=raw),root)

    def test_association_supported_and_sparse_cases(self):
        rng=np.random.default_rng(22)
        n=800
        d=pd.DataFrame(dict(station_id=[str(i//8) for i in range(n)],validation_scheme='wse_stratified',
            primary_cohort=True,gauge_days=337,gauge_Q50=rng.lognormal(4,1,n),
            sword_width_m=rng.lognormal(5,1,n),nearest_node_distance_km=rng.uniform(0,8,n),
            longitude=rng.uniform(80,130,n),latitude=rng.uniform(20,50,n),Q_observed=rng.lognormal(4,1,n),
            error10=rng.random(n)<.3))
        d=assign_groups(d)
        with tempfile.TemporaryDirectory() as tmp:
            association_analysis(d,Path(tmp))
            status=pd.read_csv(Path(tmp)/'association_model_support.csv')
            self.assertTrue(status.status.eq('ok').all())
            out=pd.read_csv(Path(tmp)/'adjusted_associations.csv')
            self.assertTrue(np.isfinite(out[['odds_ratio','ci_low','ci_high']]).all().all())
            d['error10']=False;d.loc[0,'error10']=True
            association_analysis(d,Path(tmp))
            status=pd.read_csv(Path(tmp)/'association_model_support.csv')
            self.assertTrue(status.status.eq('insufficient_events_or_clusters').all())

if __name__=='__main__':unittest.main()
