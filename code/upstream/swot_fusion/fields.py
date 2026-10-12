"""Hydrocron field profiles.

Lists follow the official Hydrocron field catalogue.  Key and provenance fields
are repeated in each request group so large responses can be split and merged
without losing observation identity.
"""

from __future__ import annotations

from collections.abc import Iterable


NODE_KEYS = ["reach_id", "node_id", "time_str", "cycle_id", "pass_id"]
REACH_KEYS = ["reach_id", "time_str", "cycle_id", "pass_id"]

PROVENANCE_FIELDS = [
    "continent_id", "range_start_time", "range_end_time", "crid",
    "sword_version", "collection_shortname", "collection_version",
    "granuleUR", "ingest_time",
]

NODE_MEASUREMENT_FIELDS = [
    "time", "time_tai", "lat", "lon", "lat_u", "lon_u", "river_name",
    "wse", "wse_u", "wse_r_u", "wse_sm", "wse_sm_u", "wse_sm_q",
    "wse_sm_q_b", "width", "width_u", "area_total", "area_tot_u",
    "area_detct", "area_det_u", "area_wse", "layovr_val", "node_dist",
    "xtrk_dist", "flow_angle", "node_q", "node_q_b", "dark_frac",
    "ice_clim_f", "ice_dyn_f", "n_good_pix", "xovr_cal_q", "rdr_sig0",
    "rdr_sig0_u", "rdr_pol",
]

NODE_CORRECTION_PRIOR_FIELDS = [
    "geoid_hght", "solid_tide", "load_tidef", "load_tideg", "pole_tide",
    "dry_trop_c", "wet_trop_c", "iono_c", "xovr_cal_c", "p_wse",
    "p_wse_var", "p_width", "p_wid_var", "p_dist_out", "p_length",
    "p_dam_id", "p_n_ch_max", "p_n_ch_mod",
]

REACH_HYDRAULIC_FIELDS = [
    "time", "time_tai", "p_lat", "p_lon", "river_name", "wse", "wse_u",
    "wse_r_u", "wse_c", "wse_c_u", "slope", "slope_u", "slope_r_u",
    "slope2", "slope2_u", "slope2_r_u", "width", "width_u", "width_c",
    "width_c_u", "area_total", "area_tot_u", "area_detct", "area_det_u",
    "area_wse", "d_x_area", "d_x_area_u", "layovr_val", "node_dist",
    "loc_offset", "xtrk_dist", "reach_q", "reach_q_b", "dark_frac",
    "ice_clim_f", "ice_dyn_f", "partial_f", "n_good_nod", "obs_frac_n",
    "xovr_cal_q",
]

# RiverSP discharge: 14 flow-law estimates, uncertainties, scale factors and
# per-method quality codes, plus bitwise aggregate discharge quality.
REACH_DISCHARGE_FIELDS = [
    "dschg_c", "dschg_c_u", "dschg_csf", "dschg_c_q",
    "dschg_gc", "dschg_gc_u", "dschg_gcsf", "dschg_gc_q",
    "dschg_m", "dschg_m_u", "dschg_msf", "dschg_m_q",
    "dschg_gm", "dschg_gm_u", "dschg_gmsf", "dschg_gm_q",
    "dschg_b", "dschg_b_u", "dschg_bsf", "dschg_b_q",
    "dschg_gb", "dschg_gb_u", "dschg_gbsf", "dschg_gb_q",
    "dschg_h", "dschg_h_u", "dschg_hsf", "dschg_h_q",
    "dschg_gh", "dschg_gh_u", "dschg_ghsf", "dschg_gh_q",
    "dschg_o", "dschg_o_u", "dschg_osf", "dschg_o_q",
    "dschg_go", "dschg_go_u", "dschg_gosf", "dschg_go_q",
    "dschg_s", "dschg_s_u", "dschg_ssf", "dschg_s_q",
    "dschg_gs", "dschg_gs_u", "dschg_gssf", "dschg_gs_q",
    "dschg_i", "dschg_i_u", "dschg_isf", "dschg_i_q",
    "dschg_gi", "dschg_gi_u", "dschg_gisf", "dschg_gi_q",
    "dschg_q_b", "dschg_gq_b",
]

REACH_CORRECTION_NETWORK_PRIOR_FIELDS = [
    "geoid_hght", "geoid_slop", "solid_tide", "load_tidef", "load_tideg",
    "pole_tide", "dry_trop_c", "wet_trop_c", "iono_c", "xovr_cal_c",
    "n_reach_up", "n_reach_dn", "rch_id_up", "rch_id_dn", "p_wse",
    "p_wse_var", "p_width", "p_wid_var", "p_n_nodes", "p_dist_out",
    "p_length", "p_maf", "p_dam_id", "p_n_ch_max", "p_n_ch_mod",
    "p_low_slp",
]

SOS_DISCHARGE_FIELDS = [
    "sos_consensus_q", "sos_hivdi_q", "sos_metroman_q", "sos_momma_q",
    "sos_sad_q", "sos_sic4dvar_q", "sos_lakeflow_q",
    "swot_discharge_reanalysis",
]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def node_field_groups() -> list[list[str]]:
    return [
        _unique(NODE_KEYS + NODE_MEASUREMENT_FIELDS),
        _unique(NODE_KEYS + NODE_CORRECTION_PRIOR_FIELDS + PROVENANCE_FIELDS),
    ]


def reach_d_field_groups() -> list[list[str]]:
    return [
        _unique(REACH_KEYS + REACH_HYDRAULIC_FIELDS),
        _unique(REACH_KEYS + REACH_DISCHARGE_FIELDS),
        _unique(REACH_KEYS + REACH_CORRECTION_NETWORK_PRIOR_FIELDS + PROVENANCE_FIELDS),
    ]


def reach_sos_v2_field_groups() -> list[list[str]]:
    return [_unique(REACH_KEYS + SOS_DISCHARGE_FIELDS + PROVENANCE_FIELDS)]


COLLECTIONS = {
    "node_d": "SWOT_L2_HR_RiverSP_node_D",
    "reach_d": "SWOT_L2_HR_RiverSP_reach_D",
    # Hydrocron documentation requires the Version 2.0 parent river collection
    # to expose Level-4 SoS fields on Reach requests.
    "reach_sos_v2": "SWOT_L2_HR_RiverSP_2.0",
}

