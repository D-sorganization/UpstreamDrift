import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { bindSegmentMeshes, SegmentMaterialColors } from './segmentMaterialColors';

describe('segment material overrides', () => {
  it('binds the nearest declared segment without spilling into unloaded children', () => {
    const root = new THREE.Group();
    root.name = 'torso';
    const child = new THREE.Mesh();
    child.name = 'head';
    const torso = new THREE.Mesh();
    root.add(child, torso);
    const bound = bindSegmentMeshes(root, ['torso', 'head']);
    expect(bound.torso).toEqual([torso]);
    expect(bound.head).toEqual([child]);
  });
  it('isolates shared materials, preserves opacity and restores exact originals', () => {
    const material = new THREE.MeshStandardMaterial({ color: '#123456', opacity: 0.4 });
    const first = new THREE.Mesh(new THREE.BoxGeometry(), material);
    const second = new THREE.Mesh(new THREE.BoxGeometry(), material);
    const adapter = new SegmentMaterialColors({ arbitrary: [first] });
    adapter.apply({ arbitrary: '#ff0000' });
    expect((first.material as THREE.MeshStandardMaterial).color.getHexString()).toBe('ff0000');
    expect((first.material as THREE.MeshStandardMaterial).opacity).toBe(0.4);
    expect(second.material).toBe(material);
    expect(material.color.getHexString()).toBe('123456');
    const override = first.material;
    adapter.apply({ arbitrary: '#0000ff' });
    expect(first.material).toBe(override);
    adapter.apply({});
    expect(first.material).toBe(material);
    adapter.dispose();
    adapter.dispose();
  });
  it('supports multi-material meshes and rejects invalid updates before mutation', () => {
    const originals = [new THREE.MeshStandardMaterial(), new THREE.MeshBasicMaterial()];
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(), originals);
    const adapter = new SegmentMaterialColors({ link: [mesh] });
    expect(() => adapter.apply({ link: 'invalid' })).toThrow();
    expect(mesh.material).toBe(originals);
    adapter.apply({ link: '#0000ff' });
    adapter.dispose();
    expect(mesh.material).toBe(originals);
  });
});
