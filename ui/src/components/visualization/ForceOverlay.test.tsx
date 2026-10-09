/**
 * Tests for ForceOverlay Three.js rendering (ADR-0052, #11308).
 * Uses schemas/glyph-set-examples.json as authoritative wire schema fixtures.
 */

import { describe, it, expect } from "vitest";
import * as THREE from "three";
import {
  alignYTo,
  buildForceOverlayScene,
  getCylinderEndpoints,
  getConeEndpoints,
} from "./forceOverlayScene";
import type { GlyphSetV1, ArrowGlyph, TorqueArcGlyph } from "@/types/glyphs";
import glyphSetExamples from "../../../../schemas/glyph-set-examples.json";

describe("alignYTo", () => {
  it("maps +y onto the direction for the +x case", () => {
    const quat = alignYTo([1, 0, 0]);
    const rotated = new THREE.Vector3(0, 1, 0).applyQuaternion(quat);
    expect(rotated.x).toBeCloseTo(1.0, 9);
    expect(rotated.y).toBeCloseTo(0.0, 9);
    expect(rotated.z).toBeCloseTo(0.0, 9);
  });

  it("maps +y onto the direction for the -y (antiparallel) case", () => {
    const quat = alignYTo([0, -1, 0]);
    const rotated = new THREE.Vector3(0, 1, 0).applyQuaternion(quat);
    expect(rotated.x).toBeCloseTo(0.0, 9);
    expect(rotated.y).toBeCloseTo(-1.0, 9);
    expect(rotated.z).toBeCloseTo(0.0, 9);
  });

  it("maps +y onto the direction for a diagonal case", () => {
    const dir = new THREE.Vector3(1, 1, 1).normalize();
    const quat = alignYTo([dir.x, dir.y, dir.z]);
    const rotated = new THREE.Vector3(0, 1, 0).applyQuaternion(quat);
    expect(rotated.x).toBeCloseTo(dir.x, 9);
    expect(rotated.y).toBeCloseTo(dir.y, 9);
    expect(rotated.z).toBeCloseTo(dir.z, 9);
  });

  it("returns identity quaternion for zero-length direction", () => {
    const quat = alignYTo([0, 0, 0]);
    expect(quat.x).toBe(0);
    expect(quat.y).toBe(0);
    expect(quat.z).toBe(0);
    expect(quat.w).toBe(1);
  });
});

describe("ForceOverlay scene graph from glyph-set-examples fixtures", () => {
  const cases = (
    glyphSetExamples as unknown as {
      cases: Array<{
        name: string;
        valid: boolean;
        description: string;
        data: GlyphSetV1;
      }>;
    }
  ).cases;

  for (const tc of cases) {
    it(`renders scene graph correctly for fixture case: ${tc.name}`, () => {
      const scene = buildForceOverlayScene(tc.data);
      expect(scene).not.toBeNull();
      if (!scene) return;

      const arrowGroups = scene.children.filter((c) =>
        c.name.startsWith("arrow-"),
      );
      const arcGroups = scene.children.filter((c) =>
        c.name.startsWith("torque-"),
      );

      expect(arrowGroups).toHaveLength(tc.data.arrows.length);
      expect(arcGroups).toHaveLength(tc.data.torque_arcs.length);

      // Verify each arrow: 1 cylinder shaft + 1 cone head
      tc.data.arrows.forEach((glyph: ArrowGlyph, idx: number) => {
        const group = arrowGroups[idx];
        expect(group).toBeDefined();
        expect(group.userData.kind).toBe(glyph.kind);
        expect(group.userData.label).toBe(glyph.label);

        const shaftMesh = group.children.find(
          (c) => c.name === "arrow-shaft",
        ) as THREE.Mesh;
        const headMesh = group.children.find(
          (c) => c.name === "arrow-head",
        ) as THREE.Mesh;

        expect(shaftMesh).toBeDefined();
        expect(headMesh).toBeDefined();

        // Check shaft endpoints equal tail_m and head_base_m within 1e-9
        const [tail, headBaseFromShaft] = getCylinderEndpoints(shaftMesh);
        expect(tail.x).toBeCloseTo(glyph.tail_m[0], 9);
        expect(tail.y).toBeCloseTo(glyph.tail_m[1], 9);
        expect(tail.z).toBeCloseTo(glyph.tail_m[2], 9);

        expect(headBaseFromShaft.x).toBeCloseTo(glyph.head_base_m[0], 9);
        expect(headBaseFromShaft.y).toBeCloseTo(glyph.head_base_m[1], 9);
        expect(headBaseFromShaft.z).toBeCloseTo(glyph.head_base_m[2], 9);

        // Check cone head endpoints equal head_base_m and tip_m within 1e-9
        const [headBaseFromCone, tip] = getConeEndpoints(headMesh);
        expect(headBaseFromCone.x).toBeCloseTo(glyph.head_base_m[0], 9);
        expect(headBaseFromCone.y).toBeCloseTo(glyph.head_base_m[1], 9);
        expect(headBaseFromCone.z).toBeCloseTo(glyph.head_base_m[2], 9);

        expect(tip.x).toBeCloseTo(glyph.tip_m[0], 9);
        expect(tip.y).toBeCloseTo(glyph.tip_m[1], 9);
        expect(tip.z).toBeCloseTo(glyph.tip_m[2], 9);

        // Colours equal rgba values
        const material = shaftMesh.material as THREE.MeshStandardMaterial;
        expect(material.color.r).toBeCloseTo(glyph.rgba[0], 4);
        expect(material.color.g).toBeCloseTo(glyph.rgba[1], 4);
        expect(material.color.b).toBeCloseTo(glyph.rgba[2], 4);
        expect(material.opacity).toBeCloseTo(glyph.rgba[3], 4);
      });

      // Verify each torque arc: 1 tube + 1 cone head
      tc.data.torque_arcs.forEach((glyph: TorqueArcGlyph, idx: number) => {
        const group = arcGroups[idx];
        expect(group).toBeDefined();
        expect(group.userData.kind).toBe(glyph.kind);
        expect(group.userData.label).toBe(glyph.label);

        const tubeMesh = group.children.find(
          (c) => c.name === "torque-tube",
        ) as THREE.Mesh;
        const headMesh = group.children.find(
          (c) => c.name === "torque-head",
        ) as THREE.Mesh;

        expect(tubeMesh).toBeDefined();
        expect(headMesh).toBeDefined();

        // Check cone head endpoints equal head_base_m and head_tip_m within 1e-9
        const [headBaseFromCone, headTip] = getConeEndpoints(headMesh);
        expect(headBaseFromCone.x).toBeCloseTo(glyph.head_base_m[0], 9);
        expect(headBaseFromCone.y).toBeCloseTo(glyph.head_base_m[1], 9);
        expect(headBaseFromCone.z).toBeCloseTo(glyph.head_base_m[2], 9);

        expect(headTip.x).toBeCloseTo(glyph.head_tip_m[0], 9);
        expect(headTip.y).toBeCloseTo(glyph.head_tip_m[1], 9);
        expect(headTip.z).toBeCloseTo(glyph.head_tip_m[2], 9);

        // Colours equal rgba values
        const material = headMesh.material as THREE.MeshStandardMaterial;
        expect(material.color.r).toBeCloseTo(glyph.rgba[0], 4);
        expect(material.color.g).toBeCloseTo(glyph.rgba[1], 4);
        expect(material.color.b).toBeCloseTo(glyph.rgba[2], 4);
        expect(material.opacity).toBeCloseTo(glyph.rgba[3], 4);
      });
    });
  }
});

describe("Round-trip regression & DbC boundary", () => {
  it("preserves joint_actuator kind intact through scene graph generation", () => {
    const testGlyphs: GlyphSetV1 = {
      schema_version: "glyph-set-v1",
      time_s: 1.0,
      arrows: [],
      torque_arcs: [
        {
          label: "actuator:wrist",
          kind: "joint_actuator",
          center_m: [0, 0, 0],
          axis_unit: [0, 0, 1],
          radius_m: 0.1,
          polyline_m: [
            [0.1, 0, 0],
            [0, 0.1, 0],
            [-0.1, 0, 0],
          ],
          head_tip_m: [-0.1, 0, 0],
          head_base_m: [-0.08, 0.05, 0],
          rgba: [1, 0.5, 0, 1],
          magnitude: 15.0,
          units: "N*m",
          clamped: false,
        },
      ],
      legend: {
        force_reference_n: null,
        force_reference_length_m: null,
        torque_reference_nm: 15.0,
        torque_reference_radius_m: 0.1,
        kinds_present: ["joint_actuator"],
        unavailable_labels: [],
        engine: "test",
        source_labels: [],
      },
    };

    const scene = buildForceOverlayScene(testGlyphs);
    expect(scene).not.toBeNull();
    const arcGroup = scene?.children[0];
    expect(arcGroup?.userData.kind).toBe("joint_actuator");
    expect(arcGroup?.userData.label).toBe("actuator:wrist");
  });

  it("rejects an invalid schema_version (DbC contract)", () => {
    const invalid = {
      schema_version: "wrong-version",
      time_s: 0.0,
      arrows: [],
      torque_arcs: [],
      legend: {
        force_reference_n: null,
        force_reference_length_m: null,
        torque_reference_nm: null,
        torque_reference_radius_m: null,
        kinds_present: [],
        unavailable_labels: [],
        engine: "test",
        source_labels: [],
      },
    } as unknown as GlyphSetV1;

    expect(buildForceOverlayScene(invalid)).toBeNull();
  });
});

describe("Clamped arrows (GCV-4, #11710)", () => {
  const glyph = (clamped: boolean) => ({
    label: "contact:grf_net",
    kind: "contact" as const,
    tail_m: [0, 0, 0] as [number, number, number],
    tip_m: [0, 0, 0.9] as [number, number, number],
    head_base_m: [0, 0, 0.7] as [number, number, number],
    shaft_radius_m: 0.012,
    head_radius_m: 0.03,
    rgba: [1, 0.5, 0, 1] as [number, number, number, number],
    magnitude: 2500,
    units: "N",
    clamped,
  });

  it("adds a trailing second head only for clamped arrows", async () => {
    const { buildArrowMeshGroup } = await import("./forceOverlayScene");
    const plain = buildArrowMeshGroup(glyph(false));
    const clamped = buildArrowMeshGroup(glyph(true));
    expect(plain.children.map((c) => c.name)).toEqual([
      "arrow-shaft",
      "arrow-head",
    ]);
    const names = clamped.children.map((c) => c.name);
    expect(names).toContain("arrow-clamped-head");
    const head = clamped.getObjectByName("arrow-head")!;
    const second = clamped.getObjectByName("arrow-clamped-head")!;
    expect(second.position.z).toBeLessThan(head.position.z);
  });
});
