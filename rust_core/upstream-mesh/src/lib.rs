//! # upstream-mesh — UpstreamDrift mesh kernels
//!
//! Issue #5219 / #5248: Rust collision-geometry kernels backed by `parry3d`.
//! Replaces the trimesh-heavy logic in
//! `src/shared/python/humanoid_character_builder/mesh/_cg_*.py` and de-risks
//! the OOM lineage of closed #3903 by keeping peak working set independent
//! of input triangle count for the VHACD path.
//!
//! ## Modules
//!
//! - [`convex_hull`] — quickhull (deterministic vertex set + outward triangle
//!   indices) via `parry3d::transformation::convex_hull`.
//! - [`decimation`]  — VHACD-based approximate convex decomposition (replaces
//!   the trimesh `simplify_quadric_decimation` fallback for collision-mesh
//!   simplification). Quadric-error edge-collapse decimation is *not*
//!   implemented — tracked as a follow-up.
//! - [`primitives`]  — AABB / OBB (via PCA) / sphere / cylinder / capsule
//!   fitting for collision-shape approximation.
//!
//! All kernels accept `&[[f32; 3]]` vertex slices and (where applicable)
//! `&[[u32; 3]]` triangle index slices. The Python facade in
//! `humanoid_character_builder/mesh/_cg_*.py` keeps responsibility for
//! converting to/from `trimesh.Trimesh` instances.

pub mod convex_hull;
pub mod decimation;
pub mod primitives;

pub use convex_hull::{compute_convex_hull, ConvexHullError, ConvexHullResult};
pub use decimation::{decimate_vhacd, DecimateParameters, DecimationError, DecimationResult};
pub use primitives::{
    fit_aabb, fit_capsule, fit_cylinder, fit_obb, fit_sphere, AabbFit, CapsuleFit, CylinderFit,
    ObbFit, PrimitiveError, SphereFit,
};

// ── Python bindings (feature-gated) ──────────────────────────────────────────

#[cfg(feature = "python")]
mod bindings;
