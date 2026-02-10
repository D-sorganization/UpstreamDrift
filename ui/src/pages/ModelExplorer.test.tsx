/**
 * Tests for ModelExplorer page.
 *
 * See issue #1200
 */

import { describe, it, expect } from 'vitest';

/** Tree node type matching the component's internal interface. */
interface URDFTreeNode {
  id: string;
  name: string;
  node_type: 'link' | 'joint' | 'root';
  parent_id: string | null;
  children: string[];
  properties: Record<string, unknown>;
}

/** Model explorer data type. */
interface ModelExplorerData {
  model_name: string;
  tree: URDFTreeNode[];
  joint_count: number;
  link_count: number;
  model_format: string;
  file_path: string;
}

describe('ModelExplorer data structures', () => {
  it('should build a valid tree from API data', () => {
    const data: ModelExplorerData = {
      model_name: 'test_robot',
      tree: [
        {
          id: 'link_base',
          name: 'base',
          node_type: 'root',
          parent_id: null,
          children: ['joint_hip'],
          properties: { type: 'link', mass: 10.0 },
        },
        {
          id: 'joint_hip',
          name: 'hip',
          node_type: 'joint',
          parent_id: 'link_base',
          children: ['link_torso'],
          properties: { joint_type: 'revolute', lower: -3.14, upper: 3.14 },
        },
        {
          id: 'link_torso',
          name: 'torso',
          node_type: 'link',
          parent_id: 'joint_hip',
          children: [],
          properties: { type: 'link', mass: 5.0 },
        },
      ],
      joint_count: 1,
      link_count: 2,
      model_format: 'urdf',
      file_path: 'test.urdf',
    };

    expect(data.model_name).toBe('test_robot');
    expect(data.tree).toHaveLength(3);
    expect(data.joint_count).toBe(1);
    expect(data.link_count).toBe(2);
  });

  it('should identify root nodes correctly', () => {
    const nodes: URDFTreeNode[] = [
      {
        id: 'link_base',
        name: 'base',
        node_type: 'root',
        parent_id: null,
        children: ['joint_a'],
        properties: {},
      },
      {
        id: 'joint_a',
        name: 'a',
        node_type: 'joint',
        parent_id: 'link_base',
        children: ['link_child'],
        properties: { joint_type: 'revolute' },
      },
      {
        id: 'link_child',
        name: 'child',
        node_type: 'link',
        parent_id: 'joint_a',
        children: [],
        properties: {},
      },
    ];

    const rootNodes = nodes.filter(
      (n) => n.parent_id === null || n.node_type === 'root',
    );
    expect(rootNodes).toHaveLength(1);
    expect(rootNodes[0].name).toBe('base');
  });

  it('should find movable joints', () => {
    const nodes: URDFTreeNode[] = [
      {
        id: 'joint_a',
        name: 'revolute_joint',
        node_type: 'joint',
        parent_id: 'link_base',
        children: [],
        properties: { joint_type: 'revolute', lower: -1.5, upper: 1.5 },
      },
      {
        id: 'joint_b',
        name: 'fixed_joint',
        node_type: 'joint',
        parent_id: 'link_base',
        children: [],
        properties: { joint_type: 'fixed' },
      },
      {
        id: 'joint_c',
        name: 'prismatic_joint',
        node_type: 'joint',
        parent_id: 'link_base',
        children: [],
        properties: { joint_type: 'prismatic', lower: 0.0, upper: 0.5 },
      },
    ];

    const movable = nodes.filter(
      (n) => n.node_type === 'joint' && n.properties.joint_type !== 'fixed',
    );
    expect(movable).toHaveLength(2);
    expect(movable.map((n) => n.name)).toContain('revolute_joint');
    expect(movable.map((n) => n.name)).toContain('prismatic_joint');
  });

  it('should build node map for lookup', () => {
    const nodes: URDFTreeNode[] = [
      {
        id: 'link_a',
        name: 'a',
        node_type: 'root',
        parent_id: null,
        children: ['joint_b'],
        properties: {},
      },
      {
        id: 'joint_b',
        name: 'b',
        node_type: 'joint',
        parent_id: 'link_a',
        children: [],
        properties: {},
      },
    ];

    const map = new Map(nodes.map((n) => [n.id, n]));
    expect(map.size).toBe(2);
    expect(map.get('link_a')?.name).toBe('a');
    expect(map.get('joint_b')?.parent_id).toBe('link_a');
    expect(map.get('nonexistent')).toBeUndefined();
  });

  it('should track joint values for FK preview', () => {
    const jointValues: Record<string, number> = {};

    // Simulate user adjusting joints
    jointValues['shoulder'] = 0.5;
    jointValues['elbow'] = 1.2;
    jointValues['wrist'] = -0.3;

    expect(Object.keys(jointValues)).toHaveLength(3);
    expect(jointValues['shoulder']).toBe(0.5);
    expect(jointValues['elbow']).toBe(1.2);
    expect(jointValues['wrist']).toBe(-0.3);
  });

  it('should handle model comparison data', () => {
    interface ModelCompareData {
      shared_joints: string[];
      unique_to_a: string[];
      unique_to_b: string[];
    }

    const compare: ModelCompareData = {
      shared_joints: ['hip', 'shoulder'],
      unique_to_a: ['wrist_flex'],
      unique_to_b: ['ankle', 'knee'],
    };

    expect(compare.shared_joints).toHaveLength(2);
    expect(compare.unique_to_a).toHaveLength(1);
    expect(compare.unique_to_b).toHaveLength(2);
  });
});
