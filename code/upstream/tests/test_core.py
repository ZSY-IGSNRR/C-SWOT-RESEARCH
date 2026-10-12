from __future__ import annotations

import json
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from swot_fusion.fields import (
    PROVENANCE_FIELDS,
    REACH_DISCHARGE_FIELDS,
    SOS_DISCHARGE_FIELDS,
    reach_d_field_groups,
)
from swot_fusion.config import load_config
from swot_fusion.fusion import (
    _attach_reach_slope_fallback,
    _filter_station_overpass_hampel,
    _limit_station_nodes,
)
from swot_fusion.hydrocron import (
    _request_with_retry_and_split,
    merge_field_parts,
    parse_hydrocron_csv,
)
from swot_fusion.quality import quality_control_frame


class CoreTests(unittest.TestCase):
    def test_default_inventory_is_capped_at_three_nodes(self) -> None:
        config = load_config()
        self.assertEqual(config["inventory"]["node_neighbors_each_side"], 1)
        mapping = pd.DataFrame({
            "node_id": [str(i) for i in range(7)],
            "neighbor_offset": [-3, -2, -1, 0, 1, 2, 3],
        })
        limited = _limit_station_nodes(mapping, config)
        self.assertEqual(set(limited["neighbor_offset"]), {-1, 0, 1})

    def test_parse_direct_and_json_encoded_csv(self) -> None:
        csv_text = "reach_id,time_str,wse\n1,2024-01-01T00:00:00Z,2.5\n"
        self.assertEqual(len(parse_hydrocron_csv(csv_text)), 1)
        self.assertEqual(len(parse_hydrocron_csv(json.dumps(csv_text))), 1)

    def test_hydrocron_feature_not_found_is_no_data(self) -> None:
        error = HTTPError(
            url="https://example.invalid",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=BytesIO(
                b'{"error":"400: Results with the specified Feature ID 1 were not found"}'
            ),
        )
        with patch("swot_fusion.hydrocron._request_once", side_effect=error):
            result = _request_with_retry_and_split(
                feature="Node",
                feature_id="1",
                collection_name="collection",
                fields=["node_id", "time_str"],
                start_time="2023-03-01T00:00:00Z",
                end_time="2026-09-03T23:59:59Z",
                timeout=1,
                api_key=None,
                retries=0,
                backoff=0,
            )
        self.assertTrue(result.empty)

    def test_merge_field_parts_uses_observation_keys(self) -> None:
        first = pd.DataFrame({
            "reach_id": ["1"], "time_str": ["2024-01-01T00:00:00Z"],
            "cycle_id": [1], "pass_id": [2], "wse": [3.0],
        })
        second = pd.DataFrame({
            "reach_id": ["1"], "time_str": ["2024-01-01T00:00:00Z"],
            "cycle_id": [1], "pass_id": [2], "dschg_c": [10.0],
        })
        merged = merge_field_parts([first, second], "Reach")
        self.assertEqual(merged.loc[0, "wse"], 3.0)
        self.assertEqual(merged.loc[0, "dschg_c"], 10.0)

    def test_merge_field_parts_collapses_duplicate_observation_keys(self) -> None:
        first = pd.DataFrame({
            "reach_id": ["1", "1"],
            "time_str": ["2024-01-01T00:00:00Z"] * 2,
            "cycle_id": [1, 1],
            "pass_id": [2, 2],
            "wse": [3.0, 5.0],
        })
        second = pd.DataFrame({
            "reach_id": ["1"],
            "time_str": ["2024-01-01T00:00:00Z"],
            "cycle_id": [1],
            "pass_id": [2],
            "dschg_c": [10.0],
        })
        merged = merge_field_parts([first, second], "Reach")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.loc[0, "wse"], 4.0)
        self.assertEqual(merged.loc[0, "dschg_c"], 10.0)

    def test_all_riversp_discharge_families_and_provenance_are_requested(self) -> None:
        fields = {field for group in reach_d_field_groups() for field in group}
        self.assertLessEqual(set(REACH_DISCHARGE_FIELDS), fields)
        self.assertLessEqual(set(PROVENANCE_FIELDS), fields)
        self.assertEqual(len(SOS_DISCHARGE_FIELDS), 8)

    def test_quality_control_preserves_suspect_and_rejects_bad(self) -> None:
        frame = pd.DataFrame({
            "node_id": ["1", "1", "1"],
            "reach_id": ["2", "2", "2"],
            "time_str": ["2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z"],
            "node_q": [0, 1, 3], "wse": [1.0, 2.0, -999999999999.0], "width": [5.0, 6.0, 7.0],
        })
        config = {
            "quality": {
                "accepted_node_q": [0, 1], "accepted_reach_q": [0, 1],
                "accepted_discharge_q": [0, 1], "exclude_hampel_outliers": False,
                "hampel_window": 3, "hampel_sigma": 3.0,
            }
        }
        clean, audit = quality_control_frame(frame, "node_d", config)
        self.assertEqual(len(clean), 2)
        self.assertEqual(audit["rows_quality_rejected"], 1)
        self.assertTrue(np.isfinite(clean["wse"]).all())

    def test_station_overpass_hampel_removes_common_wse_spike(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="21D", tz="UTC")
        frame = pd.DataFrame({
            "time_utc": dates,
            "pass_id": [1] * 7,
            "wse": [10.0, 10.1, 9.9, 50.0, 10.0, 10.1, 9.9],
            "width": [100.0, 101.0, 99.0, 100.0, 100.0, 101.0, 99.0],
        })
        config = load_config()
        clean, audit = _filter_station_overpass_hampel(frame, config)
        self.assertEqual(audit["overpasses_station_hampel_flagged"], 1)
        self.assertEqual(len(clean), 6)
        self.assertLess(clean["wse"].max(), 20.0)

    def test_wide_time_fallback_changes_only_slope(self) -> None:
        merged = pd.DataFrame({
            "time_utc": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-22T00:00:00Z"], utc=True
            ),
            "reach_reach_time_utc": pd.to_datetime(
                ["2024-01-01T00:00:00Z", None], utc=True
            ),
            "reach_match_delta_seconds": [0.0, np.nan],
            "reach_slope": [0.001, np.nan],
            "reach_dschg_c": [10.0, np.nan],
        })
        reach = pd.DataFrame({
            "reach_reach_time_utc": pd.to_datetime(
                ["2024-01-01T00:00:00Z"], utc=True
            ),
            "reach_slope": [0.001],
            "reach_dschg_c": [10.0],
        })
        out = _attach_reach_slope_fallback(
            merged, reach, tolerance=pd.Timedelta(days=21)
        )
        self.assertAlmostEqual(out.loc[1, "reach_slope"], 0.001)
        self.assertTrue(out.loc[1, "reach_slope_fallback_used"])
        self.assertTrue(pd.isna(out.loc[1, "reach_dschg_c"]))


if __name__ == "__main__":
    unittest.main()
