import type { URDFModel, URDFLinkGeometry, URDFJointDescriptor } from '@/components/visualization/URDFViewer';

export interface URDFTreeNode {
  id: string;
  name: string;
  node_type: 'link' | 'joint' | 'root';
  parent_id: string | null;
  children: string[];
  properties: Record<string, unknown>;
}

export interface TreeDiff {
  added: string[];
  removed: string[];
  modified: Array<{
    id: string;
    changes: Array<{
      field: string;
      sourceVal: unknown;
      targetVal: unknown;
    }>;
  }>;
}

function createUniqueSwapId(baseId: string, usedIds: Set<string>): string {
  let newId = baseId;
  let counter = 1;
  while (usedIds.has(newId)) {
    newId = `${baseId}_swap_${counter}`;
    counter++;
  }
  usedIds.add(newId);
  return newId;
}

function remapSubtreeForDestination(
  subtreeNodes: URDFTreeNode[],
  destinationNodes: URDFTreeNode[],
  rootNodeId: string,
  destinationParentId: string | null,
): { nodes: URDFTreeNode[]; rootId: string } {
  const usedIds = new Set(destinationNodes.map((node) => node.id));
  const idMap = new Map<string, string>();

  for (const node of subtreeNodes) {
    idMap.set(node.id, createUniqueSwapId(node.id, usedIds));
  }

  return {
    rootId: idMap.get(rootNodeId)!,
    nodes: subtreeNodes.map((node) => ({
      ...node,
      id: idMap.get(node.id)!,
      parent_id:
        node.id === rootNodeId
          ? destinationParentId
          : node.parent_id
            ? idMap.get(node.parent_id)!
            : null,
      children: node.children.map((childId) => idMap.get(childId)!),
      properties: { ...node.properties },
    })),
  };
}

/**
 * Copies a single component from the source tree (without children)
 * and inserts it under targetParentId in targetTree.
 */
export function copyComponent(
  sourceTree: URDFTreeNode[],
  targetTree: URDFTreeNode[],
  sourceNodeId: string,
  targetParentId: string | null,
): URDFTreeNode[] {
  const sourceNode = sourceTree.find((n) => n.id === sourceNodeId);
  if (!sourceNode) return targetTree;

  // Clone the target tree
  const newTargetTree = targetTree.map((n) => ({
    ...n,
    children: [...n.children],
    properties: { ...n.properties },
  }));

  // Create unique ID
  let newId = `${sourceNodeId}_copy`;
  let counter = 1;
  while (newTargetTree.some((n) => n.id === newId)) {
    newId = `${sourceNodeId}_copy_${counter}`;
    counter++;
  }

  const newNode: URDFTreeNode = {
    ...sourceNode,
    id: newId,
    parent_id: targetParentId,
    children: [], // Only single component is copied
    properties: { ...sourceNode.properties },
  };

  newTargetTree.push(newNode);

  if (targetParentId) {
    const parent = newTargetTree.find((n) => n.id === targetParentId);
    if (parent && !parent.children.includes(newId)) {
      parent.children.push(newId);
    }
  }

  return newTargetTree;
}

/**
 * Copies a component and all of its descendants recursively from
 * sourceTree to targetTree and inserts it under targetParentId.
 */
export function copyLinkChain(
  sourceTree: URDFTreeNode[],
  targetTree: URDFTreeNode[],
  sourceNodeId: string,
  targetParentId: string | null,
): URDFTreeNode[] {
  const sourceNodeMap = new Map(sourceTree.map((n) => [n.id, n]));
  const newTargetTree = targetTree.map((n) => ({
    ...n,
    children: [...n.children],
    properties: { ...n.properties },
  }));

  // Helper function to recursively copy a subtree
  function copySubtree(nodeId: string, currentTargetParentId: string | null): string | null {
    const sNode = sourceNodeMap.get(nodeId);
    if (!sNode) return null;

    let newId = `${nodeId}_copy`;
    let counter = 1;
    while (newTargetTree.some((n) => n.id === newId)) {
      newId = `${nodeId}_copy_${counter}`;
      counter++;
    }

    const newNode: URDFTreeNode = {
      ...sNode,
      id: newId,
      parent_id: currentTargetParentId,
      children: [], // will be populated
      properties: { ...sNode.properties },
    };

    newTargetTree.push(newNode);

    // Recursively copy children
    for (const childId of sNode.children) {
      const copiedChildId = copySubtree(childId, newId);
      if (copiedChildId) {
        newNode.children.push(copiedChildId);
      }
    }

    return newId;
  }

  const rootCopiedId = copySubtree(sourceNodeId, targetParentId);

  if (rootCopiedId && targetParentId) {
    const parent = newTargetTree.find((n) => n.id === targetParentId);
    if (parent && !parent.children.includes(rootCopiedId)) {
      parent.children.push(rootCopiedId);
    }
  }

  return newTargetTree;
}

/**
 * Swaps two subtrees between source and target trees.
 */
export function swapSubtrees(
  sourceTree: URDFTreeNode[],
  targetTree: URDFTreeNode[],
  sourceNodeId: string,
  targetNodeId: string,
): { sourceTree: URDFTreeNode[]; targetTree: URDFTreeNode[] } {
  const sourceNodeMap = new Map(sourceTree.map((n) => [n.id, n]));
  const targetNodeMap = new Map(targetTree.map((n) => [n.id, n]));

  const sourceNode = sourceNodeMap.get(sourceNodeId);
  const targetNode = targetNodeMap.get(targetNodeId);

  if (!sourceNode || !targetNode) {
    return { sourceTree, targetTree };
  }

  const sourceParentId = sourceNode.parent_id;
  const targetParentId = targetNode.parent_id;

  // Collect descendants
  function collectDescendants(nodeId: string, map: Map<string, URDFTreeNode>): Set<string> {
    const set = new Set<string>([nodeId]);
    const queue = [nodeId];
    while (queue.length > 0) {
      const curr = queue.shift()!;
      const n = map.get(curr);
      if (n) {
        for (const childId of n.children) {
          if (!set.has(childId)) {
            set.add(childId);
            queue.push(childId);
          }
        }
      }
    }
    return set;
  }

  const sourceSubtreeIds = collectDescendants(sourceNodeId, sourceNodeMap);
  const targetSubtreeIds = collectDescendants(targetNodeId, targetNodeMap);

  // Separate nodes
  const sourceSubtreeNodes = sourceTree.filter((n) => sourceSubtreeIds.has(n.id));
  const sourceRemainingNodes = sourceTree.filter((n) => !sourceSubtreeIds.has(n.id));

  const targetSubtreeNodes = targetTree.filter((n) => targetSubtreeIds.has(n.id));
  const targetRemainingNodes = targetTree.filter((n) => !targetSubtreeIds.has(n.id));
  const remappedTargetSubtree = remapSubtreeForDestination(
    targetSubtreeNodes,
    sourceRemainingNodes,
    targetNodeId,
    sourceParentId,
  );
  const remappedSourceSubtree = remapSubtreeForDestination(
    sourceSubtreeNodes,
    targetRemainingNodes,
    sourceNodeId,
    targetParentId,
  );

  // Map and modify trees
  const newSourceTree = [
    ...sourceRemainingNodes.map((n) => {
      if (n.id === sourceParentId) {
        return {
          ...n,
          children: n.children.map((cid) => (
            cid === sourceNodeId ? remappedTargetSubtree.rootId : cid
          )),
        };
      }
      return n;
    }),
    ...remappedTargetSubtree.nodes,
  ];

  const newTargetTree = [
    ...targetRemainingNodes.map((n) => {
      if (n.id === targetParentId) {
        return {
          ...n,
          children: n.children.map((cid) => (
            cid === targetNodeId ? remappedSourceSubtree.rootId : cid
          )),
        };
      }
      return n;
    }),
    ...remappedSourceSubtree.nodes,
  ];

  return {
    sourceTree: newSourceTree,
    targetTree: newTargetTree,
  };
}

/**
 * Merges all nodes from sourceTree into targetTree under targetParentId.
 */
export function mergeTrees(
  sourceTree: URDFTreeNode[],
  targetTree: URDFTreeNode[],
  targetParentId: string | null,
): URDFTreeNode[] {
  const sourceRoots = sourceTree.filter((n) => n.parent_id === null || n.node_type === 'root');
  let currentTargetTree = targetTree;
  for (const root of sourceRoots) {
    currentTargetTree = copyLinkChain(sourceTree, currentTargetTree, root.id, targetParentId);
  }
  return currentTargetTree;
}

/**
 * Computes differences between sourceTree and targetTree.
 */
export function computeTreeDiff(
  sourceTree: URDFTreeNode[],
  targetTree: URDFTreeNode[],
): TreeDiff {
  const sourceMap = new Map(sourceTree.map((n) => [n.id, n]));
  const targetMap = new Map(targetTree.map((n) => [n.id, n]));

  const added: string[] = [];
  const removed: string[] = [];
  const modified: TreeDiff['modified'] = [];

  for (const [id, targetNode] of targetMap.entries()) {
    if (!sourceMap.has(id)) {
      added.push(id);
    } else {
      const sourceNode = sourceMap.get(id)!;
      const changes: Array<{ field: string; sourceVal: unknown; targetVal: unknown }> = [];

      if (sourceNode.name !== targetNode.name) {
        changes.push({ field: 'name', sourceVal: sourceNode.name, targetVal: targetNode.name });
      }
      if (sourceNode.node_type !== targetNode.node_type) {
        changes.push({ field: 'node_type', sourceVal: sourceNode.node_type, targetVal: targetNode.node_type });
      }
      if (sourceNode.parent_id !== targetNode.parent_id) {
        changes.push({ field: 'parent_id', sourceVal: sourceNode.parent_id, targetVal: targetNode.parent_id });
      }

      const srcChildren = [...sourceNode.children].sort();
      const tgtChildren = [...targetNode.children].sort();
      if (JSON.stringify(srcChildren) !== JSON.stringify(tgtChildren)) {
        changes.push({ field: 'children', sourceVal: sourceNode.children, targetVal: targetNode.children });
      }

      const allPropKeys = new Set([
        ...Object.keys(sourceNode.properties),
        ...Object.keys(targetNode.properties),
      ]);
      for (const key of allPropKeys) {
        const srcVal = sourceNode.properties[key];
        const tgtVal = targetNode.properties[key];
        if (JSON.stringify(srcVal) !== JSON.stringify(tgtVal)) {
          changes.push({ field: key, sourceVal: srcVal, targetVal: tgtVal });
        }
      }

      if (changes.length > 0) {
        modified.push({ id, changes });
      }
    }
  }

  for (const id of sourceMap.keys()) {
    if (!targetMap.has(id)) {
      removed.push(id);
    }
  }

  return { added, removed, modified };
}

/**
 * Converts a Frankenstein tree node array into a URDFModel structure suitable for
 * 3D canvas preview rendering in URDFViewer.
 */
export function treeToURDFModel(
  treeNodes: URDFTreeNode[],
  modelName: string = 'frankenstein',
): URDFModel {
  const links: URDFLinkGeometry[] = [];
  const joints: URDFJointDescriptor[] = [];
  const childLinks = new Set<string>();

  const parseVec3 = (val: unknown, fallback: [number, number, number]): [number, number, number] => {
    if (Array.isArray(val) && val.length === 3) {
      return [Number(val[0]), Number(val[1]), Number(val[2])];
    }
    if (typeof val === 'string') {
      const parts = val.trim().split(/\s+/).map(Number);
      if (parts.length === 3 && !parts.some(isNaN)) {
        return [parts[0], parts[1], parts[2]];
      }
    }
    return fallback;
  };

  for (const node of treeNodes) {
    if (node.node_type === 'link' || node.node_type === 'root') {
      const geomType = (node.properties.geometry_type as URDFLinkGeometry['geometry_type']) || 'box';
      const dims: Record<string, number> = {};
      if (node.properties.dimensions && typeof node.properties.dimensions === 'object') {
        Object.assign(dims, node.properties.dimensions);
      } else {
        if ('size' in node.properties && typeof node.properties.size === 'string') {
          const parts = node.properties.size.split(/\s+/).map(Number);
          dims.width = parts[0] ?? 0.1;
          dims.height = parts[1] ?? 0.1;
          dims.depth = parts[2] ?? 0.1;
        } else {
          if ('width' in node.properties) dims.width = Number(node.properties.width);
          if ('height' in node.properties) dims.height = Number(node.properties.height);
          if ('depth' in node.properties) dims.depth = Number(node.properties.depth);
          if ('radius' in node.properties) dims.radius = Number(node.properties.radius);
          if ('length' in node.properties) dims.length = Number(node.properties.length);
        }
      }

      links.push({
        link_name: node.name,
        geometry_type: geomType,
        dimensions: dims,
        origin: parseVec3(node.properties.origin ?? node.properties.xyz, [0, 0, 0]),
        rotation: parseVec3(node.properties.rotation ?? node.properties.rpy, [0, 0, 0]),
        color: (node.properties.color as [number, number, number, number]) || [0.5, 0.5, 0.5, 1.0],
        mesh_path: typeof node.properties.mesh_path === 'string' ? node.properties.mesh_path : null,
      });
    } else if (node.node_type === 'joint') {
      const parentLink = String(node.properties.parent_link || '');
      const childLink = String(node.properties.child_link || '');
      if (childLink) {
        childLinks.add(childLink);
      }

      joints.push({
        name: node.name,
        joint_type: (node.properties.joint_type as URDFJointDescriptor['joint_type']) || 'fixed',
        parent_link: parentLink,
        child_link: childLink,
        origin: parseVec3(node.properties.xyz ?? node.properties.origin, [0, 0, 0]),
        rotation: parseVec3(node.properties.rpy ?? node.properties.rotation, [0, 0, 0]),
        axis: parseVec3(node.properties.axis, [0, 0, 1]),
        lower_limit: typeof node.properties.lower === 'number' ? node.properties.lower : null,
        upper_limit: typeof node.properties.upper === 'number' ? node.properties.upper : null,
      });
    }
  }

  let rootLink = 'base';
  const rootNode = treeNodes.find((n) => n.node_type === 'root');
  if (rootNode) {
    rootLink = rootNode.name;
  } else {
    const unparentedLink = links.find((l) => !childLinks.has(l.link_name));
    if (unparentedLink) {
      rootLink = unparentedLink.link_name;
    } else if (links.length > 0) {
      rootLink = links[0].link_name;
    }
  }

  return {
    model_name: modelName,
    links,
    joints,
    root_link: rootLink,
  };
}

