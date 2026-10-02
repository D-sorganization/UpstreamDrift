import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LibraryActions } from './LibraryActions';

const api = vi.hoisted(() => ({createPlayer: vi.fn(), createSwing: vi.fn(), importAsset: vi.fn()}));
vi.mock('@/api/necromatcher', () => api);
beforeEach(() => { Object.values(api).forEach((mock) => mock.mockReset()); });

describe('Historical Library Actions', () => {
  it('does not carry a swing draft to a different selected player', async () => {
    const user = userEvent.setup();
    const changed = vi.fn();
    const view = render(<LibraryActions player="ben-hogan" swing="" onChanged={changed} />);
    await user.type(screen.getByLabelText('Swing ID'), 'hogan-drive');
    await user.type(screen.getByLabelText('Swing Name'), 'Hogan Drive');
    view.rerender(<LibraryActions player="tiger-woods" swing="" onChanged={changed} />);
    expect(screen.getByLabelText('Swing ID')).toHaveValue('');
    expect(screen.getByLabelText('Swing Name')).toHaveValue('');
  });
  it('retains failed values for correction and refreshes only after a successful retry', async () => {
    api.createSwing.mockRejectedValueOnce(new Error('Swing ID already exists')).mockResolvedValueOnce({});
    const user = userEvent.setup();
    const changed = vi.fn();
    render(<LibraryActions player="ben-hogan" swing="" onChanged={changed} />);
    await user.type(screen.getByLabelText('Swing ID'), 'drive');
    await user.type(screen.getByLabelText('Swing Name'), 'Drive');
    await user.click(screen.getByRole('button', {name: 'Save Swing'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('Swing ID already exists');
    expect(screen.getByLabelText('Swing ID')).toHaveValue('drive');
    expect(changed).not.toHaveBeenCalled();
    await user.clear(screen.getByLabelText('Swing ID'));
    await user.type(screen.getByLabelText('Swing ID'), 'drive-v2');
    await user.click(screen.getByRole('button', {name: 'Save Swing'}));
    await waitFor(() => expect(changed).toHaveBeenCalledOnce());
    expect(api.createSwing).toHaveBeenLastCalledWith('drive-v2', 'ben-hogan', 'Drive');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
  it('does not import a previous swing draft into the newly recalled swing', async () => {
    const user = userEvent.setup();
    const changed = vi.fn();
    const view = render(<LibraryActions player="ben-hogan" swing="hogan-drive" onChanged={changed} />);
    await user.type(screen.getByLabelText('Version ID'), 'hogan-capture-v1');
    await user.type(screen.getByLabelText('Source Path'), 'C:/captures/hogan');
    view.rerender(<LibraryActions player="tiger-woods" swing="tiger-drive" onChanged={changed} />);
    expect(screen.getByLabelText('Version ID')).toHaveValue('');
    expect(screen.getByLabelText('Source Path')).toHaveValue('');
    expect(api.importAsset).not.toHaveBeenCalled();
  });
  it('submits ordered model joints once while saving and preserves the request snapshot', async () => {
    let finish: (value: unknown) => void = () => {};
    api.importAsset.mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    const user = userEvent.setup();
    const changed = vi.fn();
    render(<LibraryActions player="ben-hogan" swing="hogan-drive" onChanged={changed} />);
    await user.selectOptions(screen.getByLabelText('Import Type'), 'models');
    await user.type(screen.getByLabelText('Ordered Joint Names'), ' hip, knee, shoulder ');
    await user.type(screen.getByLabelText('Version ID'), 'model-v1');
    await user.type(screen.getByLabelText('Source Path'), 'C:/captures/model.xml');
    await user.click(screen.getByRole('button', {name: 'Import Version'}));
    expect(screen.getByRole('button', {name: 'Saving…'})).toBeDisabled();
    await user.selectOptions(screen.getByLabelText('Model Engine'), 'drake');
    expect(api.importAsset).toHaveBeenCalledExactlyOnceWith('hogan-drive', 'models', {
      id: 'model-v1', source_path: 'C:/captures/model.xml', engine: 'mujoco', dofs: ['hip', 'knee', 'shoulder'],
    });
    finish({});
    await waitFor(() => expect(changed).toHaveBeenCalledOnce());
  });
});
