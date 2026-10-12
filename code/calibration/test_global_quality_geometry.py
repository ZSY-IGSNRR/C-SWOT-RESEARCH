import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import quality_geometry as q
import run_beta_wse0_width0_experiments as exp
import swot_beta_wse0_width0_model as core
import run_review_revision_experiments as pipeline
from common_support import paired_comparisons


def frame(n=30):
    return pd.DataFrame({'station_id':['x']*n,'obs_id':[str(i) for i in range(n)],
        'time_dt':pd.date_range('2024-01-01',periods=n,tz='UTC'),'wse_model':100+np.arange(n)*.1,
        'width':20+np.arange(n),'slope':np.full(n,1e-4),'Q':10+np.arange(n,dtype=float)})


class QualityGeometryTest(unittest.TestCase):
    def setUp(self):q.configure(SimpleNamespace());exp._install_overrides()

    def test_global_filter_catches_validation_spike_and_zero_mad(self):
        a=frame();a['width']=20.;a.loc[15,'width']=2000
        a=q.annotate(a)
        self.assertFalse(a.loc[15,'global_qc_pass'])
        pairs,_=exp.make_split_frame_versions(a,[str(i) for i in range(15)],['15','16'])
        self.assertEqual(pairs['full'][1].obs_id.tolist(),['16'])
        self.assertEqual(pairs['no_hampel'][1].obs_id.tolist(),['15','16'])

    def test_qc_does_not_read_discharge(self):
        a=frame();b=a.copy();b['Q']=np.nan
        columns=['global_qc_pass','global_qc_reason']
        pd.testing.assert_frame_equal(q.annotate(a)[columns],q.annotate(b)[columns])

    def test_endpoint_spike_is_checked(self):
        bad,_,_=q.hampel_flags([1000,10,10,10,10,10],1)
        self.assertTrue(bad[0])

    def test_invalid_slope_is_shared_basic_filter(self):
        a=frame();a.loc[0,'slope']=0;a=q.annotate(a)
        self.assertNotIn('0',q.retain(a,'full').obs_id.tolist())
        self.assertNotIn('0',q.retain(a,'no_hampel').obs_id.tolist())

    def test_initial_width_requires_local_evidence(self):
        a=frame(4);a['width']=[1000,20,22,24]
        guarded=q.curve(a,growth=False)
        self.assertTrue(guarded['first_width_corrected'])
        self.assertLess(guarded['curve_width'][0],30)
        a['width']=[1000,1100,1200,1300]
        self.assertFalse(q.curve(a)['first_width_corrected'])
        a['wse_model']=[0,100,200,300];a['width']=[1000,20,22,24]
        self.assertFalse(q.curve(a)['first_width_corrected'])

    def test_width_growth_is_bounded(self):
        a=frame(4);a['wse_model']=[0,.1,1,2];a['width']=[10,1000,10000,100000]
        model=q.curve(a,initial=False)
        x,y=model['curve_wse'],model['curve_width']
        self.assertTrue(np.all(np.diff(y)>=0))
        self.assertTrue(np.all(np.diff(y)<=np.minimum(100,2*y[:-1])*np.diff(x)+1e-9))
        self.assertGreater(q.curve(a,initial=False,growth=False)['curve_width'][-1],y[-1])

    def test_validation_width_never_refits_geometry(self):
        a=q.annotate(frame());b=a.copy();b.loc[20:,'width']=1e7
        ids=[str(i) for i in range(20)];test=['20','21','22']
        pa,_=exp.make_split_frame_versions(a,ids,test);pb,_=exp.make_split_frame_versions(b,ids,test)
        np.testing.assert_allclose(pa['full'][0].width_model,pb['full'][0].width_model)
        np.testing.assert_allclose(pa['full'][1].width_model,pb['full'][1].width_model)
        self.assertTrue((pb['width_train_only'][1].width_model==1e7).all())
        np.testing.assert_allclose(pa['full'][0].width_model,pa['width_train_only'][0].width_model)

    def test_temporal_fraction_and_expanding_windows(self):
        a=frame(40);folds=core.temporal_forward_splits(a,3,.7)
        self.assertEqual(len(folds[0][1]),28)
        tests=[]
        for _,train,test in folds:
            self.assertLess(train.time_dt.max(),test.time_dt.min())
            tests.extend(test.obs_id)
        self.assertEqual(len(tests),12);self.assertEqual(len(set(tests)),12)

    def test_paired_comparison_uses_same_observations(self):
        rows=[]
        for method in ['full','no_hampel']:
            for i in range(6 if method=='full' else 7):
                rows.append({'station_id':'x','repeat_id':0,'validation_scheme':'wse_stratified',
                    'method':method,'obs_id':str(i),'Q_observed':i+1.,'Q_calculated':i+1. if i<6 else 1e9})
        cfg=SimpleNamespace(ablation_repeat_count=1,seed=42,bootstrap_iterations=200)
        raw,summary,metrics,coverage=paired_comparisons(pd.DataFrame(rows),cfg,pipeline)
        self.assertTrue(raw.full_advantage.eq(0).all())
        self.assertEqual(int(coverage.common_count.iloc[0]),6)
        self.assertEqual(int(coverage.comparator_native_count.iloc[0]),7)


if __name__=='__main__':unittest.main()
