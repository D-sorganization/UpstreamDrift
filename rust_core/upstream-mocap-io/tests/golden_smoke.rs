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
