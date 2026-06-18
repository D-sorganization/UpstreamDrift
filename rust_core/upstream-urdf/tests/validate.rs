//! Domain-validation tests for the strict URDF parser (issue #7659).
//!
//! `parse_urdf_str` must reject malformed numeric attributes and physically
//! invalid robots (non-positive mass, non-positive-definite inertia,
//! `lower > upper` joint limits) with a typed error instead of silently
//! substituting defaults. `parse_urdf_str_lenient` performs the same
//! structural parse but skips the domain checks.

use upstream_urdf::{parse_urdf_str, parse_urdf_str_lenient, UrdfError};

/// A structurally valid, physically valid single-link/single-joint robot.
const VALID: &str = r#"<?xml version="1.0"?>
<robot name="r">
  <link name="base">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>
  <link name="tip"/>
  <joint name="j" type="revolute">
    <parent link="base"/>
    <child link="tip"/>
    <limit lower="-1.0" upper="1.0" effort="10" velocity="1"/>
  </joint>
</robot>"#;

#[test]
fn valid_urdf_still_parses() {
    let r = parse_urdf_str(VALID).expect("valid URDF must parse");
    assert_eq!(r.name, "r");
    assert_eq!(r.links.len(), 2);
    assert_eq!(r.joints.len(), 1);
}

#[test]
fn malformed_numeric_attribute_rejected() {
    // `mass="abc"` previously silently became 0.0; now it must error.
    let xml = VALID.replace(r#"<mass value="1.0"/>"#, r#"<mass value="abc"/>"#);
    let err = parse_urdf_str(&xml).unwrap_err();
    assert!(
        matches!(err, UrdfError::Parse(_)),
        "expected Parse error for malformed mass, got {err:?}"
    );
}

#[test]
fn malformed_limit_attribute_rejected() {
    let xml = VALID.replace(r#"lower="-1.0""#, r#"lower="x""#);
    let err = parse_urdf_str(&xml).unwrap_err();
    assert!(matches!(err, UrdfError::Parse(_)), "got {err:?}");
}

#[test]
fn non_positive_mass_rejected() {
    let xml = VALID.replace(r#"<mass value="1.0"/>"#, r#"<mass value="0.0"/>"#);
    let err = parse_urdf_str(&xml).unwrap_err();
    assert!(matches!(err, UrdfError::Schema(_)), "got {err:?}");

    let xml = VALID.replace(r#"<mass value="1.0"/>"#, r#"<mass value="-2.0"/>"#);
    let err = parse_urdf_str(&xml).unwrap_err();
    assert!(matches!(err, UrdfError::Schema(_)), "got {err:?}");
}

#[test]
fn non_positive_definite_inertia_rejected() {
    // izz = 0 → non-positive diagonal.
    let xml = VALID.replace(r#"izz="0.05""#, r#"izz="0.0""#);
    let err = parse_urdf_str(&xml).unwrap_err();
    assert!(matches!(err, UrdfError::Schema(_)), "got {err:?}");

    // Large off-diagonal ixy breaks SPD even with positive diagonal.
    let xml = VALID.replace(r#"ixy="0.0""#, r#"ixy="1.0""#);
    let err = parse_urdf_str(&xml).unwrap_err();
    assert!(matches!(err, UrdfError::Schema(_)), "got {err:?}");
}

#[test]
fn inverted_joint_limits_rejected() {
    let xml = VALID.replace(r#"lower="-1.0" upper="1.0""#, r#"lower="1.0" upper="-1.0""#);
    let err = parse_urdf_str(&xml).unwrap_err();
    assert!(matches!(err, UrdfError::Schema(_)), "got {err:?}");
}

#[test]
fn negative_effort_or_velocity_rejected() {
    let xml = VALID.replace(r#"effort="10""#, r#"effort="-10""#);
    let err = parse_urdf_str(&xml).unwrap_err();
    assert!(matches!(err, UrdfError::Schema(_)), "got {err:?}");

    let xml = VALID.replace(r#"velocity="1""#, r#"velocity="-1""#);
    let err = parse_urdf_str(&xml).unwrap_err();
    assert!(matches!(err, UrdfError::Schema(_)), "got {err:?}");
}

#[test]
fn lenient_parser_accepts_physically_invalid_but_well_formed_robot() {
    // Zero mass is a domain violation but structurally well-formed: the lenient
    // parser must still accept it (malformed numbers are still rejected).
    let xml = VALID.replace(r#"<mass value="1.0"/>"#, r#"<mass value="0.0"/>"#);
    let r = parse_urdf_str_lenient(&xml).expect("lenient parse ignores domain rules");
    assert_eq!(r.links[0].inertial.as_ref().unwrap().mass, 0.0);

    // But a malformed numeric attribute is still a hard error even in lenient
    // mode.
    let bad = VALID.replace(r#"<mass value="1.0"/>"#, r#"<mass value="abc"/>"#);
    assert!(parse_urdf_str_lenient(&bad).is_err());
}
