/**
 * Three.js scene graph builder for force/torque overlay glyphs (ADR-0052, #11308).
 * Pure geometric mapping with no physics and no client-side scaling/clamping maths.
 */

import * as THREE from 'three';
import type { GlyphSetV1, ArrowGlyph, TorqueArcGlyph } from '@/types/glyphs';

/**
 * Computes a quaternion rotating the +Y unit vector [0, 1, 0] to the target direction.
 * If direction is antiparallel [0, -1, 0], rotates by PI around the X axis.
 */
export function alignYTo(
  direction: THREE.Vector3 | [number, number, number],
): THREE.Quaternion {
  const dir = Array.isArray(direction)
    ? new THREE.Vector3(direction[0], direction[1], direction[2]).normalize()
    : direction.clone().normalize();

  const up = new THREE.Vector3(0, 1, 0);
  const quat = new THREE.Quaternion();

  if (dir.lengthSq() < 1e-12) {
    return quat;
  }

  const dot = up.dot(dir);
  if (dot < -0.9999999) {
    return quat.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
  }

  return quat.setFromUnitVectors(up, dir);
}

/**
 * Extracts the bottom and top endpoints of a cylinder mesh in world/scene coordinates.
 */
export function getCylinderEndpoints(mesh: THREE.Mesh): [THREE.Vector3, THREE.Vector3] {
  const geo = mesh.geometry as THREE.CylinderGeometry;
  const height = geo.parameters.height;
  const bottom = new THREE.Vector3(0, -height / 2, 0).applyQuaternion(mesh.quaternion).add(mesh.position);
  const top = new THREE.Vector3(0, height / 2, 0).applyQuaternion(mesh.quaternion).add(mesh.position);
  return [bottom, top];
}

/**
 * Extracts the base and tip endpoints of a cone mesh in world/scene coordinates.
 */
export function getConeEndpoints(mesh: THREE.Mesh): [THREE.Vector3, THREE.Vector3] {
  const geo = mesh.geometry as THREE.ConeGeometry;
  const height = geo.parameters.height;
  const base = new THREE.Vector3(0, -height / 2, 0).applyQuaternion(mesh.quaternion).add(mesh.position);
  const tip = new THREE.Vector3(0, height / 2, 0).applyQuaternion(mesh.quaternion).add(mesh.position);
  return [base, tip];
}

export function buildArrowMeshGroup(glyph: ArrowGlyph): THREE.Group {
  const group = new THREE.Group();
  group.name = `arrow-${glyph.label}`;
  group.userData = { kind: glyph.kind, label: glyph.label };

  const tail = new THREE.Vector3(...glyph.tail_m);
  const headBase = new THREE.Vector3(...glyph.head_base_m);
  const tip = new THREE.Vector3(...glyph.tip_m);

  const shaftLen = tail.distanceTo(headBase);
  const shaftMid = tail.clone().add(headBase).multiplyScalar(0.5);
  const shaftDir = headBase.clone().sub(tail).normalize();
  const shaftQuat = alignYTo(shaftDir);

  const headLen = headBase.distanceTo(tip);
  const headMid = headBase.clone().add(tip).multiplyScalar(0.5);
  const headDir = tip.clone().sub(headBase).normalize();
  const headQuat = alignYTo(headDir);

  const color = new THREE.Color(glyph.rgba[0], glyph.rgba[1], glyph.rgba[2]);
  const opacity = glyph.rgba[3];
  const material = new THREE.MeshStandardMaterial({
    color,
    transparent: opacity < 1.0,
    opacity,
  });

  // Shaft cylinder
  const shaftGeo = new THREE.CylinderGeometry(
    glyph.shaft_radius_m,
    glyph.shaft_radius_m,
    shaftLen,
    16,
  );
  const shaftMesh = new THREE.Mesh(shaftGeo, material);
  shaftMesh.name = 'arrow-shaft';
  shaftMesh.position.copy(shaftMid);
  shaftMesh.quaternion.copy(shaftQuat);
  group.add(shaftMesh);

  // Head cone
  const headGeo = new THREE.ConeGeometry(glyph.head_radius_m, headLen, 16);
  const headMesh = new THREE.Mesh(headGeo, material);
  headMesh.name = 'arrow-head';
  headMesh.position.copy(headMid);
  headMesh.quaternion.copy(headQuat);
  group.add(headMesh);

  // Clamped arrows (raw length above the style maximum) get a trailing second
  // head so a shortened arrow never reads as a smaller force (GCV-4, #11710).
  if (glyph.clamped) {
    const chevron = new THREE.Mesh(headGeo, material);
    chevron.name = 'arrow-clamped-head';
    chevron.position.copy(headMid).addScaledVector(headDir, -0.6 * headLen);
    chevron.quaternion.copy(headQuat);
    group.add(chevron);
  }

  return group;
}

export function buildTorqueArcMeshGroup(glyph: TorqueArcGlyph): THREE.Group {
  const group = new THREE.Group();
  group.name = `torque-${glyph.label}`;
  group.userData = { kind: glyph.kind, label: glyph.label };

  const points = glyph.polyline_m.map((pt) => new THREE.Vector3(...pt));
  const curve = new THREE.CatmullRomCurve3(points, false, 'centripetal');
  const tubeRadius = Math.max(0.002, glyph.radius_m * 0.04);
  const tubeGeo = new THREE.TubeGeometry(
    curve,
    Math.max(32, points.length * 2),
    tubeRadius,
    8,
    false,
  );

  const headBase = new THREE.Vector3(...glyph.head_base_m);
  const headTip = new THREE.Vector3(...glyph.head_tip_m);
  const headLen = headBase.distanceTo(headTip);
  const headMid = headBase.clone().add(headTip).multiplyScalar(0.5);
  const headDir = headTip.clone().sub(headBase).normalize();
  const headQuat = alignYTo(headDir);
  const headRadius = headLen * 0.25;

  const color = new THREE.Color(glyph.rgba[0], glyph.rgba[1], glyph.rgba[2]);
  const opacity = glyph.rgba[3];
  const material = new THREE.MeshStandardMaterial({
    color,
    transparent: opacity < 1.0,
    opacity,
  });

  const tubeMesh = new THREE.Mesh(tubeGeo, material);
  tubeMesh.name = 'torque-tube';
  group.add(tubeMesh);

  const headGeo = new THREE.ConeGeometry(headRadius, headLen, 16);
  const headMesh = new THREE.Mesh(headGeo, material);
  headMesh.name = 'torque-head';
  headMesh.position.copy(headMid);
  headMesh.quaternion.copy(headQuat);
  group.add(headMesh);

  return group;
}

/**
 * Builds the complete Three.js scene graph for a given GlyphSetV1 payload.
 * Rejects payloads with unrecognized schema_version (DbC contract).
 */
export function buildForceOverlayScene(glyphs?: GlyphSetV1 | null): THREE.Group | null {
  if (!glyphs) return null;
  if (glyphs.schema_version !== 'glyph-set-v1') {
    return null;
  }

  const rootGroup = new THREE.Group();
  rootGroup.name = 'force-overlay';

  if (Array.isArray(glyphs.arrows)) {
    for (const arrow of glyphs.arrows) {
      rootGroup.add(buildArrowMeshGroup(arrow));
    }
  }

  if (Array.isArray(glyphs.torque_arcs)) {
    for (const arc of glyphs.torque_arcs) {
      rootGroup.add(buildTorqueArcMeshGroup(arc));
    }
  }

  return rootGroup;
}
