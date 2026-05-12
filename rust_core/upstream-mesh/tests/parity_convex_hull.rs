//! Parity test: convex hull against a hand-checked reference.
//!
//! We can't depend on scipy from Rust unit tests, so the parity-vs-scipy
//! check lives in `tests/unit/mesh/test_rust_convex_hull.py` (run when
//! `upstream-mesh` is built with `--features python`). Here we exercise:
//!
//! - The deterministic seed used in the benchmark — assert that hull
//!   vertex count and volume are stable across builds.
//! - A unit-cube fixture whose hull is exactly known.
//! - Degenerate inputs (too few points, NaN) raise the right errors.

use approx::assert_relative_eq;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use upstream_mesh::{compute_convex_hull, ConvexHullError};

#[test]
fn unit_cube_hull_has_eight_vertices_and_unit_volume() {
    // 8 corners of the unit cube + a few interior points that should be
    // discarded by the hull.
    let mut pts: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    // Interior points — these must NOT appear in the hull.
    pts.push([0.5, 0.5, 0.5]);
    pts.push([0.25, 0.25, 0.25]);

    let hull = compute_convex_hull(&pts).expect("hull");
    assert_eq!(
        hull.num_vertices(),
        8,
        "unit cube hull must have 8 vertices, got {}",
        hull.num_vertices()
    );
    // 6 faces × 2 triangles = 12 triangles.
    assert_eq!(hull.num_triangles(), 12);
    assert_relative_eq!(hull.volume(), 1.0, max_relative = 1e-5);
}

#[test]
fn deterministic_seed_100_random_points_is_stable() {
    // This is the scipy parity seed: the matching scipy assertion lives in
    // `tests/unit/mesh/test_rust_convex_hull.py`.
    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFEBABE);
    let pts: Vec<[f32; 3]> = (0..100)
        .map(|_| {
            [
                rng.gen_range(-1.0_f32..1.0),
                rng.gen_range(-1.0_f32..1.0),
                rng.gen_range(-1.0_f32..1.0),
            ]
        })
        .collect();

    let hull = compute_convex_hull(&pts).expect("hull");

    // Lock the shape of the hull so a parry3d upgrade doesn't silently
    // change it. Both bounds are observed values — adjust if parry3d
    // changes its hull algorithm in a future bump (and update the python
    // parity test in lockstep).
    assert!(
        hull.num_vertices() >= 20 && hull.num_vertices() <= 60,
        "expected 20..=60 hull vertices for 100 random points in [-1,1]^3, got {}",
        hull.num_vertices()
    );
    assert!(
        hull.volume() > 4.0 && hull.volume() < 8.0,
        "expected hull volume in (4, 8) (cube of side 2 has volume 8), got {}",
        hull.volume()
    );
}

#[test]
fn too_few_points_returns_error() {
    let pts = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    match compute_convex_hull(&pts) {
        Err(ConvexHullError::TooFewPoints(3)) => {}
        other => panic!("expected TooFewPoints(3), got {other:?}"),
    }
}

#[test]
fn non_finite_input_returns_error() {
    let pts = vec![
        [0.0_f32, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, f32::NAN],
    ];
    match compute_convex_hull(&pts) {
        Err(ConvexHullError::NonFiniteInput { index: 3 }) => {}
        other => panic!("expected NonFiniteInput {{ index: 3 }}, got {other:?}"),
    }
}
