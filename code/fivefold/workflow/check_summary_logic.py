"""Small in-memory checks for summary definitions; writes no observations."""
import numpy as np
import pandas as pd
import summarize_fivefold as s


def main():
    rows = []
    for station in ["s1", "s2"]:
        for j in range(25):
            obs = 10. + j + (100. if station == "s2" else 0.)
            for k, method in enumerate(s.METHODS):
                wse = float(j)
                rows.append(dict(station_id=station, obs_id=f"{station}_{j}",
                    repeat_id=0, fold_id=j % 5, WSE=wse, Q_observed=obs,
                    Q_calculated=obs * (1 + k / 100) + (-1.) ** j,
                    method=method, train_wse_min=2., train_wse_max=22.,
                    domain="below" if wse < 2 else "above" if wse > 22 else "within"))
    new = pd.DataFrame(rows)
    full = new[new.method.eq("full")]
    m = s.metrics(full)
    err = full.Q_calculated.to_numpy() - full.Q_observed.to_numpy()
    assert np.isclose(m["RMSE"], np.sqrt(np.mean(err ** 2)))
    assert np.isclose(m["NSE"], 1 - (err @ err) / np.sum((full.Q_observed - full.Q_observed.mean()) ** 2))
    assert np.isclose(m["SST"], m["within_gauge_SST"] + m["between_gauge_SST"])
    q = s.distribution([0, 1, 2, 3, 100])
    assert q["median"] == 2 and q["half_IQR"] == 1
    out = s.aggregate_scheme(new, "wse_stratified_5fold", True)
    assert not out["station_unique_observation_metrics"].empty
    national = out["cross_gauge_metrics_by_split"]
    selected = national[national.support.eq("native") & national.domain.eq("all") & national.method.eq("full")]
    assert len(selected) == 5 and selected.nobs.eq(10).all()
    uu = out["unique_observation_and_stacked_metrics"]
    assert len(uu[uu.support.eq("native")]) == 13 * 5
    assert uu[uu.support.eq("native") & uu.domain.eq("all")].nobs.eq(50).all()
    station = out["station_unique_observation_metrics"]
    assert station[station.support.eq("native") & station.domain.eq("all")].nobs.eq(25).all()
    gains = s.paired_gains(national)
    assert len(gains) == 5 * 3 * 2
    extreme = out["extreme_residual_metrics_by_split"]
    e = extreme[extreme.support.eq("native") & extreme.domain.eq("all")]
    assert e.n_removed.eq(1).all() and e.nobs.eq(9).all()
    legacy = s.legacy_station_medians(out["station_metrics_by_split_legacy"])
    assert not s.station_distributions(legacy, legacy=True).empty
    old_parts = []
    for repeat in range(10):
        old_parts.append(new.assign(repeat_id=repeat, fold_id=0))
    old = pd.concat(old_parts, ignore_index=True)
    old_out = s.aggregate_scheme(old, "wse_stratified_10_repeats", False)
    old_unique = old_out["unique_observation_and_stacked_metrics"]
    med = old_unique[old_unique.support.eq("native") & old_unique.domain.eq("all") &
        old_unique.aggregation.eq("median_of_available_repeated_validation_predictions_sensitivity")]
    stacked = old_unique[old_unique.support.eq("native") & old_unique.domain.eq("all") &
        old_unique.aggregation.eq("literal_stacked_repeated_predictions_sensitivity")]
    assert med.nobs.eq(50).all() and stacked.nobs.eq(500).all()
    inter, cover = s.intersection_sensitivity(new, old)
    assert len(inter) == 13 * 2 * 5 * (5 + 10)
    assert cover[cover.domain.eq("all")].n_unique_observations.eq(50).all()
    assert all(not (s.FORBIDDEN & set(frame.columns)) for frame in out.values())
    assignment = full[s.KEY].copy().assign(qc_group="pass")
    original_read = s.read
    try:
        s.read = lambda _path: assignment.copy()
        complete = s.validate_assignments(__file__, new)
        assert complete["full_model_complete_coverage"] and complete["missing_full_predictions"] == 0
        one_missing = new.drop(full.index[:1])
        incomplete = s.validate_assignments(__file__, one_missing)
        assert not incomplete["full_model_complete_coverage"] and incomplete["missing_full_predictions"] == 1
    finally:
        s.read = original_read
    print("PASS: independent metric identities; unique five-fold OOF; 13 methods; shared supports; paired gains; exact 1% removal; legacy summaries; old/new intersections; no row-level output fields.")


if __name__ == "__main__":
    main()
