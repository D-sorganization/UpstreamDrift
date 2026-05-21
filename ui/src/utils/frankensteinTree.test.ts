import { describe, it, expect } from 'vitest';
import {
  copyComponent,
  copyLinkChain,
  swapSubtrees,
  mergeTrees,
  computeTreeDiff,
  URDFTreeNode,
} from './frankensteinTree';

describe('Frankenstein URDF Tree Operations', () => {
  const sourceTree: URDFTreeNode[] = [
    {
      id: 'src_root',
      name: 'src_root',
      node_type: 'root',
      parent_id: null,
      children: ['src_joint_1'],
      properties: { mass: 10 },
    },
    {
      id: 'src_joint_1',
      name: 'src_joint_1',
      node_type: 'joint',
      parent_id: 'src_root',
      children: ['src_link_1'],
      properties: { joint_type: 'revolute' },
    },
    {
      id: 'src_link_1',
      name: 'src_link_1',
      node_type: 'link',
      parent_id: 'src_joint_1',
      children: [],
      properties: { mass: 2 },
    },
  ];

  const targetTree: URDFTreeNode[] = [
    {
      id: 'tgt_root',
      name: 'tgt_root',
      node_type: 'root',
      parent_id: null,
      children: ['tgt_joint_1'],
      properties: { mass: 12 },
    },
    {
      id: 'tgt_joint_1',
      name: 'tgt_joint_1',
      node_type: 'joint',
      parent_id: 'tgt_root',
      children: ['tgt_link_1'],
      properties: { joint_type: 'fixed' },
    },
    {
      id: 'tgt_link_1',
      name: 'tgt_link_1',
      node_type: 'link',
      parent_id: 'tgt_joint_1',
      children: [],
      properties: { mass: 3 },
    },
  ];

  it('should copy a single component without children and insert it under target parent', () => {
    // Copy src_joint_1 (joint) under tgt_link_1 (link)
    const result = copyComponent(sourceTree, targetTree, 'src_joint_1', 'tgt_link_1');

    const copiedNode = result.find((n) => n.id === 'src_joint_1_copy');
    expect(copiedNode).toBeDefined();
    expect(copiedNode?.parent_id).toBe('tgt_link_1');
    expect(copiedNode?.children).toHaveLength(0); // single component, so no children copied
    expect(copiedNode?.properties.joint_type).toBe('revolute');

    const parentNode = result.find((n) => n.id === 'tgt_link_1');
    expect(parentNode?.children).toContain('src_joint_1_copy');
  });

  it('should copy a link chain recursively and insert it under target parent', () => {
    // Copy src_joint_1 and its children recursively under tgt_link_1
    const result = copyLinkChain(sourceTree, targetTree, 'src_joint_1', 'tgt_link_1');

    const copiedJoint = result.find((n) => n.id === 'src_joint_1_copy');
    const copiedLink = result.find((n) => n.id === 'src_link_1_copy');

    expect(copiedJoint).toBeDefined();
    expect(copiedLink).toBeDefined();

    expect(copiedJoint?.parent_id).toBe('tgt_link_1');
    expect(copiedJoint?.children).toContain('src_link_1_copy');
    expect(copiedLink?.parent_id).toBe('src_joint_1_copy');

    const parentNode = result.find((n) => n.id === 'tgt_link_1');
    expect(parentNode?.children).toContain('src_joint_1_copy');
  });

  it('should swap subtrees between source and target trees', () => {
    // Swap src_joint_1 subtree (src_joint_1 -> src_link_1) with tgt_joint_1 subtree (tgt_joint_1 -> tgt_link_1)
    const { sourceTree: newSrc, targetTree: newTgt } = swapSubtrees(
      sourceTree,
      targetTree,
      'src_joint_1',
      'tgt_joint_1',
    );

    const srcRoot = newSrc.find((n) => n.id === 'src_root');
    expect(srcRoot?.children).toContain('tgt_joint_1');
    expect(srcRoot?.children).not.toContain('src_joint_1');

    const tgtRoot = newTgt.find((n) => n.id === 'tgt_root');
    expect(tgtRoot?.children).toContain('src_joint_1');
    expect(tgtRoot?.children).not.toContain('tgt_joint_1');

    const movedSrcJoint = newTgt.find((n) => n.id === 'src_joint_1');
    expect(movedSrcJoint?.parent_id).toBe('tgt_root');

    const movedTgtJoint = newSrc.find((n) => n.id === 'tgt_joint_1');
    expect(movedTgtJoint?.parent_id).toBe('src_root');
  });

  it('remaps swapped subtree ids when the destination already uses them', () => {
    const overlappingSourceTree: URDFTreeNode[] = [
      {
        id: 'src_root',
        name: 'src_root',
        node_type: 'root',
        parent_id: null,
        children: ['src_joint_1', 'shared_link'],
        properties: {},
      },
      {
        id: 'src_joint_1',
        name: 'src_joint_1',
        node_type: 'joint',
        parent_id: 'src_root',
        children: ['src_link_1'],
        properties: {},
      },
      {
        id: 'src_link_1',
        name: 'src_link_1',
        node_type: 'link',
        parent_id: 'src_joint_1',
        children: [],
        properties: {},
      },
      {
        id: 'shared_link',
        name: 'shared_link',
        node_type: 'link',
        parent_id: 'src_root',
        children: [],
        properties: {},
      },
    ];

    const overlappingTargetTree: URDFTreeNode[] = [
      {
        id: 'tgt_root',
        name: 'tgt_root',
        node_type: 'root',
        parent_id: null,
        children: ['tgt_joint_1', 'src_link_1'],
        properties: {},
      },
      {
        id: 'tgt_joint_1',
        name: 'tgt_joint_1',
        node_type: 'joint',
        parent_id: 'tgt_root',
        children: ['shared_link'],
        properties: {},
      },
      {
        id: 'shared_link',
        name: 'shared_link',
        node_type: 'link',
        parent_id: 'tgt_joint_1',
        children: [],
        properties: {},
      },
      {
        id: 'src_link_1',
        name: 'src_link_1',
        node_type: 'link',
        parent_id: 'tgt_root',
        children: [],
        properties: {},
      },
    ];

    const { sourceTree: newSrc, targetTree: newTgt } = swapSubtrees(
      overlappingSourceTree,
      overlappingTargetTree,
      'src_joint_1',
      'tgt_joint_1',
    );

    expect(new Set(newSrc.map((n) => n.id)).size).toBe(newSrc.length);
    expect(new Set(newTgt.map((n) => n.id)).size).toBe(newTgt.length);

    const remappedIntoSource = newSrc.find(
      (n) => n.parent_id === 'src_root' && n.id !== 'shared_link',
    );
    expect(remappedIntoSource).toBeDefined();
    expect(remappedIntoSource?.children).toHaveLength(1);
    expect(remappedIntoSource?.children[0]).not.toBe('shared_link');

    const remappedChildIntoSource = newSrc.find(
      (n) => n.parent_id === remappedIntoSource?.id,
    );
    expect(remappedChildIntoSource?.id).not.toBe('shared_link');

    const remappedIntoTarget = newTgt.find(
      (n) => n.parent_id === 'tgt_root' && n.id !== 'src_link_1',
    );
    expect(remappedIntoTarget).toBeDefined();
    expect(remappedIntoTarget?.children[0]).not.toBe('src_link_1');

    const remappedChildIntoTarget = newTgt.find(
      (n) => n.parent_id === remappedIntoTarget?.id,
    );
    expect(remappedChildIntoTarget?.id).not.toBe('src_link_1');
  });

  it('should merge all nodes from source into target under target parent', () => {
    const result = mergeTrees(sourceTree, targetTree, 'tgt_link_1');

    // All source nodes should exist in target tree, renamed to avoid collisions
    const rootCopy = result.find((n) => n.id === 'src_root_copy');
    const jointCopy = result.find((n) => n.id === 'src_joint_1_copy');
    const linkCopy = result.find((n) => n.id === 'src_link_1_copy');

    expect(rootCopy).toBeDefined();
    expect(jointCopy).toBeDefined();
    expect(linkCopy).toBeDefined();

    expect(rootCopy?.parent_id).toBe('tgt_link_1');
    expect(rootCopy?.children).toContain('src_joint_1_copy');
    expect(jointCopy?.parent_id).toBe('src_root_copy');
    expect(jointCopy?.children).toContain('src_link_1_copy');
    expect(linkCopy?.parent_id).toBe('src_joint_1_copy');

    const targetParentNode = result.find((n) => n.id === 'tgt_link_1');
    expect(targetParentNode?.children).toContain('src_root_copy');
  });

  it('should compute differences between source and target trees', () => {
    // To compare, let's create a modified target tree
    const modifiedTarget: URDFTreeNode[] = [
      {
        id: 'src_root',
        name: 'src_root',
        node_type: 'root',
        parent_id: null,
        children: ['src_joint_1'],
        properties: { mass: 15 }, // changed mass from 10 to 15
      },
      {
        id: 'src_joint_1',
        name: 'src_joint_1',
        node_type: 'joint',
        parent_id: 'src_root',
        children: [], // removed src_link_1 child
        properties: { joint_type: 'revolute' },
      },
      // src_link_1 is missing (removed)
      {
        id: 'new_link_added',
        name: 'new_link_added',
        node_type: 'link',
        parent_id: 'src_joint_1',
        children: [],
        properties: { mass: 5 }, // new node added
      },
    ];

    const diff = computeTreeDiff(sourceTree, modifiedTarget);

    expect(diff.added).toContain('new_link_added');
    expect(diff.removed).toContain('src_link_1');
    const modifiedRoot = diff.modified.find((m) => m.id === 'src_root');
    expect(modifiedRoot).toBeDefined();
    const massChange = modifiedRoot?.changes.find((c) => c.field === 'mass');
    expect(massChange?.sourceVal).toBe(10);
    expect(massChange?.targetVal).toBe(15);
  });
});
