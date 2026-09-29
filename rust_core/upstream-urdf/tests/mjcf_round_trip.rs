//! MJCF round-trip tests.
//!
//! Acceptance criterion for UD #5243: parse → write → parse on
//! representative MJCF fixtures (and on every `.xml` MJCF found in
//! tests/fixtures/) yields the same structural AST.
//!
//! We deliberately *do not* sweep every `.xml` under the repo because
//! many MyoSuite / OpenSim fixtures use `<include>` directives that
//! reference sibling files and complex `<default>` inheritance which
//! are out of scope for this PR (see the deferred-features section of
//! UD #5243). The synthetic fixtures here cover the 80% surface used
//! by the historical Python `MJCFConverter`.

use std::fs;
use std::path::PathBuf;

use upstream_urdf::{parse_mjcf_str, write_mjcf};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

#[test]
fn round_trip_minimal_mjcf() {
    let xml = r#"<?xml version="1.0"?>
<mujoco model="m">
  <worldbody>
    <body name="b" pos="1 2 3">
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>"#;
    let d1 = parse_mjcf_str(xml).expect("parse 1");
    assert_eq!(d1.model, "m");
    assert_eq!(d1.worldbody.bodies.len(), 1);
    assert_eq!(d1.worldbody.bodies[0].pos, [1.0, 2.0, 3.0]);
    assert_eq!(d1.worldbody.bodies[0].geoms[0].type_, "sphere");
    let written = write_mjcf(&d1).expect("write");
    let d2 = parse_mjcf_str(&written).expect("parse 2");
    assert_eq!(d1, d2, "round-trip should be structurally identical");
}

#[test]
fn round_trip_fixtures() {
    let dir = fixtures_dir();
    let entries = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("xml"))
        .collect::<Vec<_>>();
    assert!(
        !entries.is_empty(),
        "no MJCF fixtures found in {}",
        dir.display()
    );

    let mut passed = 0usize;
    let mut failures = Vec::new();
    for path in &entries {
        let xml = fs::read_to_string(path).expect("read");
        let d1 = match parse_mjcf_str(&xml) {
            Ok(d) => d,
            Err(e) => {
                failures.push(format!("{}: parse 1: {e}", path.display()));
                continue;
            }
        };
        let written = match write_mjcf(&d1) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{}: write: {e}", path.display()));
                continue;
            }
        };
        let d2 = match parse_mjcf_str(&written) {
            Ok(d) => d,
            Err(e) => {
                failures.push(format!("{}: parse 2: {e}", path.display()));
                continue;
            }
        };
        if d1 == d2 {
            passed += 1;
        } else {
            failures.push(format!(
                "{}: parse → write → parse not structurally equal",
                path.display()
            ));
        }
    }
    eprintln!(
        "MJCF round-trip: {passed} / {} fixtures passed",
        entries.len()
    );
    for f in &failures {
        eprintln!("  FAIL: {f}");
    }
    assert_eq!(passed, entries.len(), "{} failures", entries.len() - passed);
}

#[test]
fn parses_assets_and_actuators() {
    let xml = std::fs::read_to_string(fixtures_dir().join("two_link_arm.xml"))
        .expect("read two_link_arm.xml");
    let doc = parse_mjcf_str(&xml).expect("parse");
    assert_eq!(doc.model, "two_link_arm");
    assert_eq!(doc.assets.len(), 3);
    assert_eq!(doc.actuators.len(), 2);
    assert_eq!(doc.worldbody.bodies.len(), 1);
    let upper = &doc.worldbody.bodies[0];
    assert_eq!(upper.name, "upper_arm");
    assert_eq!(upper.bodies.len(), 1);
    let fore = &upper.bodies[0];
    assert_eq!(fore.name, "forearm");
    assert_eq!(fore.sites.len(), 1);
    assert_eq!(fore.sites[0].name.as_deref(), Some("end_effector"));
}

#[test]
fn benchmark_largest_fixture() {
    // Find the largest fixture and measure parse + write time. This is
    // a smoke benchmark — we just want to confirm the parser is in the
    // sub-second range on something non-trivial.
    let dir = fixtures_dir();
    let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("xml"))
        .collect();
    entries.sort_by_key(|p| fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    let Some(largest) = entries.last() else {
        return;
    };
    let xml = fs::read_to_string(largest).unwrap();
    let bytes = xml.len();
    let t0 = std::time::Instant::now();
    let doc = parse_mjcf_str(&xml).expect("parse");
    let parse_us = t0.elapsed().as_micros();
    let t1 = std::time::Instant::now();
    let _out = write_mjcf(&doc).expect("write");
    let write_us = t1.elapsed().as_micros();
    eprintln!(
        "[bench] largest fixture {} bytes={bytes} parse={parse_us}us write={write_us}us",
        largest.display()
    );
}
