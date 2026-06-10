//! Smoke tests: parse the checked-in golden files and assert basic shape.
//! Numerical parity vs the Python reference is enforced by the pytest suite
//! in `tests/unit/motion_pipeline/sources/test_mocap_io_rust_parity.py`.

use std::path::PathBuf;

use upstream_mocap_io::{bvh, c3d, trc};

fn golden(name: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
        .join("../..")
        .join("tests/data/motion_pipeline/golden")
        .join(name)
}

#[test]
fn parse_c3d_smoke() {
    let path = golden("sample.c3d");
    if !path.exists() {
        eprintln!("skipping: golden file missing at {path:?}");
        return;
    }
    let data = c3d::parse_c3d_file(&path).expect("parse sample.c3d");
    assert_eq!(data.n_frames, 30);
    assert_eq!(data.n_markers, 6);
    assert!(data.fps > 0.0);
    assert_eq!(data.positions.len(), data.n_frames * data.n_markers * 3);
}

fn repo_data(name: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest).join("../..").join("data").join(name)
}

/// The Tour-Average golf captures declare POINT:UNITS = "m" via a 1-D
/// char parameter. A parser that misses that (and falls back to the mm
/// default) shrinks the swing to a ~2 mm bounding box — regression
/// guard for the dims.len()==1 UNITS fix.
#[test]
fn parse_c3d_meter_units_ta_golf() {
    let path = repo_data("C3D_TA_Driver.c3d");
    if !path.exists() {
        eprintln!("skipping: repo data file missing at {path:?}");
        return;
    }
    let data = c3d::parse_c3d_file(&path).expect("parse C3D_TA_Driver.c3d");
    assert_eq!(data.units, "m", "TA golf files declare meters");
    assert_eq!(data.n_markers, 38);
    let max_abs = data
        .positions
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0_f32, |acc, v| acc.max(v.abs()));
    assert!(
        max_abs > 0.5 && max_abs < 10.0,
        "expected a human-scale capture in meters, got max |coord| = {max_abs}"
    );
}

/// The CMU academic captures declare POINT:UNITS = "mm"; positions must
/// come back mm→m scaled to human magnitudes.
#[test]
fn parse_c3d_mm_units_cmu() {
    let path = repo_data("cmu_mocap/subject_64/64_01.c3d");
    if !path.exists() {
        eprintln!("skipping: repo data file missing at {path:?}");
        return;
    }
    let data = c3d::parse_c3d_file(&path).expect("parse 64_01.c3d");
    assert_eq!(data.units, "mm", "CMU files declare millimeters");
    assert_eq!(data.n_markers, 45);
    let max_abs = data
        .positions
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0_f32, |acc, v| acc.max(v.abs()));
    assert!(
        max_abs > 0.5 && max_abs < 10.0,
        "expected a human-scale capture after mm→m scaling, got max |coord| = {max_abs}"
    );
}

#[test]
fn parse_trc_smoke() {
    let path = golden("sample.trc");
    if !path.exists() {
        eprintln!("skipping: golden file missing at {path:?}");
        return;
    }
    let data = trc::parse_trc_file(&path).expect("parse sample.trc");
    assert!(data.n_frames > 0);
    assert!(data.n_markers > 0);
    assert!(data.fps > 0.0);
    assert_eq!(data.positions.len(), data.n_frames * data.n_markers * 3);
}

#[test]
fn parse_bvh_smoke() {
    let path = golden("sample.bvh");
    if !path.exists() {
        eprintln!("skipping: golden file missing at {path:?}");
        return;
    }
    let data = bvh::parse_bvh_file(&path).expect("parse sample.bvh");
    assert!(data.n_frames > 0);
    assert!(!data.joints.is_empty());
    assert!(data.num_dofs > 0);
    assert!(data.fps > 0.0);
}
