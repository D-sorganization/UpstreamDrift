import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PropertyInspector, JointManipulator } from './InspectorPanel';
import { URDFTreeNode } from '@/utils/frankensteinTree';

describe('InspectorPanel Components', () => {
  describe('PropertyInspector', () => {
    it('renders placeholder when no node is selected', () => {
      render(<PropertyInspector node={null} />);
      expect(screen.getByText(/select a node to inspect/i)).toBeInTheDocument();
    });

    it('renders node details and properties', () => {
      const mockNode: URDFTreeNode = {
        id: 'node_1',
        name: 'test_link',
        node_type: 'link',
        parent_id: 'parent_1',
        children: ['child_1'],
        properties: {
          mass: 5.5,
          material: 'metal',
        },
      };

      render(<PropertyInspector node={mockNode} />);

      expect(screen.getByText('test_link')).toBeInTheDocument();
      expect(screen.getByText('link')).toBeInTheDocument();
      expect(screen.getByText('mass')).toBeInTheDocument();
      expect(screen.getByText('5.5000')).toBeInTheDocument();
      expect(screen.getByText('material')).toBeInTheDocument();
      expect(screen.getByText('metal')).toBeInTheDocument();
      expect(screen.getByText(/parent: parent_1/i)).toBeInTheDocument();
      expect(screen.getByText(/children: child_1/i)).toBeInTheDocument();
    });
  });

  describe('JointManipulator', () => {
    it('renders placeholder when no joints', () => {
      render(
        <JointManipulator
          joints={[]}
          jointValues={{}}
          onJointChange={() => {}}
        />
      );
      expect(screen.getByText(/no movable joints/i)).toBeInTheDocument();
    });

    it('renders sliders and reacts to changes', () => {
      const mockJoints: URDFTreeNode[] = [
        {
          id: 'joint_1',
          name: 'elbow',
          node_type: 'joint',
          parent_id: 'link_1',
          children: [],
          properties: {
            joint_type: 'revolute',
            lower: -1.5,
            upper: 1.5,
          },
        },
      ];

      const handleChange = vi.fn();
      render(
        <JointManipulator
          joints={mockJoints}
          jointValues={{ elbow: 0.5 }}
          onJointChange={handleChange}
        />
      );

      expect(screen.getByText('elbow')).toBeInTheDocument();
      expect(screen.getByText('0.50 rad')).toBeInTheDocument();

      const slider = screen.getByLabelText(/elbow angle/i);
      fireEvent.change(slider, { target: { value: '1.2' } });

      expect(handleChange).toHaveBeenCalledWith('elbow', 1.2);
    });
  });
});
