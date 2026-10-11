"""OpenCap augmented-marker vocabulary (#11402).

OpenCap's LSTM marker augmenter writes these 43 markers, named after the
marker sites of its LaiUhlrich2022 OpenSim model, in this order. Upstream
source of truth: ``opencap-core/opensimPipeline/Models/
LaiUhlrich2022_markers_augmenter.xml`` (Apache-2.0).

This is the single definition in UpstreamDrift. The source adapter uses it to
recognise augmented markers; ``scaling.marker_maps`` builds the
``OpenCap-LaiUhlrich2022`` marker set (segments, length pairs) from it.
"""

from __future__ import annotations

OPENCAP_MARKER_SET_NAME = "OpenCap-LaiUhlrich2022"

OPENCAP_AUGMENTED_MARKERS: tuple[str, ...] = (
    "r.ASIS_study",
    "L.ASIS_study",
    "r.PSIS_study",
    "L.PSIS_study",
    "r_knee_study",
    "r_mknee_study",
    "r_ankle_study",
    "r_mankle_study",
    "r_toe_study",
    "r_5meta_study",
    "r_calc_study",
    "L_knee_study",
    "L_mknee_study",
    "L_ankle_study",
    "L_mankle_study",
    "L_toe_study",
    "L_calc_study",
    "L_5meta_study",
    "r_shoulder_study",
    "L_shoulder_study",
    "C7_study",
    "r_lelbow_study",
    "r_melbow_study",
    "r_lwrist_study",
    "r_mwrist_study",
    "L_lelbow_study",
    "L_melbow_study",
    "L_lwrist_study",
    "L_mwrist_study",
    "r_thigh1_study",
    "r_thigh2_study",
    "r_thigh3_study",
    "L_thigh1_study",
    "L_thigh2_study",
    "L_thigh3_study",
    "r_sh1_study",
    "r_sh2_study",
    "r_sh3_study",
    "L_sh1_study",
    "L_sh2_study",
    "L_sh3_study",
    "RHJC_study",
    "LHJC_study",
)
