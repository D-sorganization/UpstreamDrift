import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ModelTree } from './ModelTree';
import { URDFTreeNode } from '@/utils/frankensteinTree';

describe('ModelTree Component', () => {
  const mockNodes: URDFTreeNode[] = [
    {
      id: 'node_root',
      name: 'root_link',
      node_type: 'root',
      parent_id: null,
      children: ['node_joint'],
      properties: {},
    },
    {
      id: 'node_joint',
      name: 'arm_joint',
      node_type: 'joint',
      parent_id: 'node_root',
      children: [],
      properties: { joint_type: 'revolute' },
    },
  ];

  it('renders all tree nodes and allows selection', () => {
    const handleSelect = vi.fn();
    render(
      <ModelTree
        modelName="Test Model"
        treeNodes={mockNodes}
        selectedNodeId={null}
        onNodeSelect={handleSelect}
        side="single"
      />
    );

    expect(screen.getByText('root_link')).toBeInTheDocument();
    // Since depth < 2 is expanded by default, arm_joint should also be rendered
    expect(screen.getByText('arm_joint')).toBeInTheDocument();

    // Click on root_link
    fireEvent.click(screen.getByText('root_link'));
    expect(handleSelect).toHaveBeenCalledWith(mockNodes[0]);
  });

  it('shows action buttons in Frankenstein mode for source side', () => {
    const handleCopyComponent = vi.fn();
    const handleCopyChain = vi.fn();

    render(
      <ModelTree
        modelName="Source Model"
        treeNodes={mockNodes}
        selectedNodeId="node_joint"
        onNodeSelect={() => {}}
        side="source"
        onCopyComponent={handleCopyComponent}
        onCopyChain={handleCopyChain}
      />
    );

    // Should show copy component/chain buttons next to or for selected node
    const copyCompBtn = screen.getByLabelText(/copy component/i);
    const copyChainBtn = screen.getByLabelText(/copy chain/i);

    expect(copyCompBtn).toBeInTheDocument();
    expect(copyChainBtn).toBeInTheDocument();

    fireEvent.click(copyCompBtn);
    expect(handleCopyComponent).toHaveBeenCalledWith('node_joint');

    fireEvent.click(copyChainBtn);
    expect(handleCopyChain).toHaveBeenCalledWith('node_joint');
  });
});
