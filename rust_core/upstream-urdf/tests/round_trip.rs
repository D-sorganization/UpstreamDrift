//! Round-trip parity tests: parse → write → parse → compare ASTs.
//!
//! Acceptance criterion for UD #5215: every URDF in `data/` (broadly
//! interpreted here as URDFs anywhere in the repo) survives a structural
//! round-trip via the Rust crate.

use std::fs;
use std::path::{Path, PathBuf};

use upstream_urdf::{parse_urdf_str, write_urdf};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is …/rust_core/upstream-urdf. Walk up two parents.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root")
        .to_path_buf()
}

fn collect_urdfs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    let skip_dirs = [
        "node_modules",
        "target",
        ".git",
        "vendor",
        "_worktrees",
        ".venv",
    ];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if skip_dirs.contains(&name) {
                        continue;
                    }
                }
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some("urdf") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

#[test]
fn round_trip_all_urdfs_in_repo() {
    let root = repo_root();
    let urdfs = collect_urdfs(&root);
    assert!(
        !urdfs.is_empty(),
        "no .urdf files found under {}",
        root.display()
    );

    let mut total = 0usize;
    let mut passed = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for path in &urdfs {
        total += 1;
        let xml = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{}: read error: {e}", path.display()));
                continue;
            }
        };
        let r1 = match parse_urdf_str(&xml) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{}: first parse: {e}", path.display()));
                continue;
            }
        };
        let written = match write_urdf(&r1) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{}: write: {e}", path.display()));
                continue;
            }
        };
        let r2 = match parse_urdf_str(&written) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{}: re-parse: {e}", path.display()));
                continue;
            }
        };
        if r1 == r2 {
            passed += 1;
        } else {
            failures.push(format!(
                "{}: parse→write→parse not structurally equal",
                path.display()
            ));
        }
    }

    eprintln!("Round-trip: {passed} / {total} URDFs passed");
    for f in &failures {
        eprintln!("  FAIL: {f}");
    }
    assert_eq!(
        passed,
        total,
        "{} of {total} round-trips failed",
        total - passed
    );
}

#[test]
fn minimal_urdf_round_trip() {
    let xml = r#"<?xml version="1.0"?>
<robot name="r">
  <link name="a"/>
  <link name="b"/>
  <joint name="j" type="revolute">
    <parent link="a"/>
    <child link="b"/>
    <origin xyz="1 2 3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>"#;
    let r1 = parse_urdf_str(xml).unwrap();
    assert_eq!(r1.name, "r");
    assert_eq!(r1.links.len(), 2);
    assert_eq!(r1.joints.len(), 1);
    assert_eq!(r1.joints[0].axis, [0.0, 1.0, 0.0]);
    let written = write_urdf(&r1).unwrap();
    let r2 = parse_urdf_str(&written).unwrap();
    assert_eq!(r1, r2);
}
