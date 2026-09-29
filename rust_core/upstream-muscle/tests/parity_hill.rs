//! Parity test: Rust Hill curves match the Python source within 1e-6.
//!
//! The fixture (`tests/parity_hill.csv`) is generated from the Python
//! source-of-truth in `src/shared/python/biomechanics/hill_muscle.py` by
//! `scripts/generate_parity_fixture.py`. Re-run that script when the
//! Python curves change.
//!
//! Slice 1 of UD#5216: covers `f_l`, `f_v`, `f_t` only. The full muscle
//! model (equilibrium solver, activation dynamics, multi-muscle moment
//! summation) lands in subsequent slices and gets an OpenSim/MuJoCo
//! parity corpus of its own.

use std::fs;
use std::path::PathBuf;

use upstream_muscle::{f_l, f_t, f_v};

const PARITY_TOL: f64 = 1e-6;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity_hill.csv")
}

fn parse_fixture(text: &str) -> Vec<(String, f64, f64)> {
    text.lines()
        .skip(1) // header
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut it = line.splitn(3, ',');
            let curve = it.next().expect("curve column").to_string();
            let input: f64 = it
                .next()
                .expect("input column")
                .parse()
                .expect("input parses as f64");
            let expected: f64 = it
                .next()
                .expect("expected column")
                .parse()
                .expect("expected parses as f64");
            (curve, input, expected)
        })
        .collect()
}

#[test]
fn rust_curves_match_python_within_1e_minus_6() {
    let text = fs::read_to_string(fixture_path()).expect("read parity fixture");
    let rows = parse_fixture(&text);
    assert!(!rows.is_empty(), "parity fixture must not be empty");

    let mut max_abs_diff: f64 = 0.0;
    let mut worst: Option<(String, f64, f64, f64)> = None;
    let (mut n_l, mut n_v, mut n_t) = (0usize, 0usize, 0usize);

    for (curve, input, expected) in rows {
        let actual = match curve.as_str() {
            "f_l" => {
                n_l += 1;
                f_l(input)
            }
            "f_v" => {
                n_v += 1;
                f_v(input)
            }
            "f_t" => {
                n_t += 1;
                f_t(input)
            }
            other => panic!("unexpected curve in fixture: {other}"),
        };
        let diff = (actual - expected).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            worst = Some((curve.clone(), input, expected, actual));
        }
    }

    assert!(
        n_l > 0 && n_v > 0 && n_t > 0,
        "fixture must cover all three curves (got f_l={n_l}, f_v={n_v}, f_t={n_t})"
    );

    if max_abs_diff > PARITY_TOL {
        panic!(
            "Hill parity exceeded tolerance {PARITY_TOL:e}: max abs diff = {max_abs_diff:e}, worst row = {worst:?}"
        );
    }

    eprintln!(
        "Hill parity OK: max abs diff = {max_abs_diff:e} across {n_l} f_l + {n_v} f_v + {n_t} f_t samples"
    );
}
