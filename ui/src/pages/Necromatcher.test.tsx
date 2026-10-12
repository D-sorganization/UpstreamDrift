import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, useNavigate, useLocation } from 'react-router';
import { NecromatcherPage } from './Necromatcher';

const mocks = vi.hoisted(() => ({ players: vi.fn(), swings: vi.fn(), assets: vi.fn(), frame: vi.fn(), projection: vi.fn(), summary: vi.fn(), plan: vi.fn(), createPlayer: vi.fn(), createSwing: vi.fn(), importAsset: vi.fn(), videoSubmit: vi.fn(), videoView: vi.fn(), videoCancel: vi.fn() }));
vi.mock('@/api/necromatcher', () => ({
  fetchPlayers: mocks.players, fetchSwings: mocks.swings, fetchAssets: mocks.assets,
  fetchCaptureFrame: mocks.frame, captureFrameImageUrl: () => '/test-source.png',
  fetchFitProjection: mocks.projection,
  fetchFitSummary: mocks.summary,
  fetchRefitPlan: mocks.plan, submitRefit: vi.fn(), fetchRefit: vi.fn(), cancelRefit: vi.fn(),
  submitVideoExport: mocks.videoSubmit, fetchVideoExport: mocks.videoView, cancelVideoExport: mocks.videoCancel, videoExportDownloadUrl: (run: string) => `/export/${run}/download`,
  swingExportUrl: (id: string) => `/api/v1/necromatcher/swings/${id}/export`,
  createPlayer: mocks.createPlayer, createSwing: mocks.createSwing, importAsset: mocks.importAsset,
}));
function show(initial = '/tools/necromatcher') {
  render(<MemoryRouter initialEntries={[initial]}><Navigation /><NecromatcherPage /></MemoryRouter>);
}
function Navigation() {
  const navigate = useNavigate();
  const location = useLocation();
  return <><output aria-label="Recalled Selection">{location.search}</output><button onClick={() => navigate('/tools/necromatcher?player=tiger-woods&swing=tiger-practice')}>Recall Tiger URL</button></>;
}
beforeEach(() => {
  Object.values(mocks).forEach((mock) => mock.mockReset());
  mocks.players.mockResolvedValue({players:[{subject_id:'ben-hogan',display_name:'Ben Hogan',metadata:{}},{subject_id:'tiger-woods',display_name:'Tiger Woods',metadata:{}}]});
  mocks.swings.mockImplementation(async (id:string) => ({swings:id==='ben-hogan'?[{session_id:'hogan-practice',subject_id:id,name:'Hogan Practice',metadata:{}}]:[{session_id:'tiger-practice',subject_id:id,name:'Tiger Practice',metadata:{}}]}));
  mocks.assets.mockResolvedValue({assets:[{dataset_id:'capture-v1',session_id:'hogan-practice',kind:'image_capture',metadata:{frame_count:3,qualification:'image_observations_only'}}]});
  mocks.frame.mockResolvedValue({capture_id:'capture-v1',frame_index:0,frame_count:3,image_width:320,image_height:240,frame:{pts_ticks:1100,timebase_numerator:1,timebase_denominator:10,physical_time_s:null},observation:{status:'detected',landmarks:{left_wrist:{x:0.5,y:0.4,visibility:null}}}});
  mocks.plan.mockImplementation(async (fit: string) => ({source_fit_id: fit, frame_indices: [0, 1, 2], coordinate_order: ['hip'], coordinate_units: ['rad'], recorded_options: null}));
  mocks.summary.mockImplementation(async (fit: string) => ({fit_id: fit,capture_id:'capture-v1',frame_count:3,frame_indices:[0,1,2]}));
});
describe('Necromatcher historical workspace', () => {
  it('renders a registered authored replay explicitly instead of controls or a kinematic fit', async () => {
    mocks.assets.mockResolvedValue({assets:[{dataset_id:'replay-v1',session_id:'hogan-practice',kind:'authored_replay',metadata:{qualification:'unqualified_authored_replay',model_id:'model'}}]});
    show('/tools/necromatcher?player=ben-hogan&swing=hogan-practice');
    expect(await screen.findByText('Authored Replay')).toBeInTheDocument();
    expect(screen.getByText('unqualified_authored_replay')).toBeInTheDocument();
    expect(screen.queryByText('Authored Controls')).not.toBeInTheDocument();
    expect(screen.queryByRole('button',{name:'Review Fit replay-v1'})).not.toBeInTheDocument();
  });
  it('reviews the saved native fit against its bound source frame', async () => {
    mocks.assets.mockResolvedValue({assets:[
      {dataset_id:'capture-v1',session_id:'hogan-practice',kind:'image_capture',metadata:{frame_count:3}},
      {dataset_id:'fit-v2',session_id:'hogan-practice',kind:'kinematic_fit',metadata:{capture_id:'capture-v1',qualification:'monocular_research_hypothesis'}},
    ]});
    mocks.projection.mockResolvedValue({fit_id:'fit-v2',capture_id:'capture-v1',frame_index:0,
      frame:{pts_ticks:1100,timebase_numerator:1,timebase_denominator:10,physical_time_s:null},
      points:{wrist:{x:160,y:96,visibility:null}},coordinates:'image_pixels',qualification:'monocular_research_hypothesis'});
    show('/tools/necromatcher?player=ben-hogan&swing=hogan-practice&capture=capture-v1&fit=fit-v2');
    expect(await screen.findByLabelText('Native Model Projection')).toBeInTheDocument();
    expect(mocks.projection).toHaveBeenCalledWith('fit-v2',0);
    expect(screen.getByText(/Camera and Physical Time Remain Unqualified/)).toBeInTheDocument();
    mocks.projection.mockImplementation(() => new Promise(() => {}));
    fireEvent.change(screen.getByRole('slider',{name:'Source Frame'}),{target:{value:'1'}});
    await waitFor(() => expect(mocks.projection).toHaveBeenCalledWith('fit-v2',1));
    expect(screen.queryByLabelText('Native Model Projection')).not.toBeInTheDocument();
  });
  it('labels saved kinematic fits as research and retains their qualification', async () => {
    mocks.assets.mockResolvedValue({assets:[{dataset_id:'hogan-fit-v2',session_id:'hogan-practice',kind:'kinematic_fit',metadata:{qualification:'monocular_research_hypothesis',model_id:'model-v2',frame_count:750}}]});
    show('/tools/necromatcher?player=ben-hogan&swing=hogan-practice');
    expect(await screen.findByText('hogan-fit-v2')).toBeInTheDocument();
    expect(screen.getAllByText('Kinematic Research Fit')).toHaveLength(2);
    expect(screen.getByText('monocular_research_hypothesis')).toBeInTheDocument();
    expect(screen.queryByText('Authored Controls')).not.toBeInTheDocument();
  });
  it('offers player tiles and retrieves the selected player swings', async () => {
    const user=userEvent.setup(); show();
    await user.click(await screen.findByRole('button',{name:'Ben Hogan'}));
    expect(await screen.findByRole('button',{name:'Hogan Practice'})).toBeInTheDocument();
    expect(mocks.swings).toHaveBeenCalledWith('ben-hogan');
    expect(screen.queryByText('Tiger Practice')).not.toBeInTheDocument();
  });
  it('recalls URL selection and displays source evidence without claiming physical time', async () => {
    show('/tools/necromatcher?player=ben-hogan&swing=hogan-practice&capture=capture-v1');
    expect(await screen.findByAltText('Historical Source Frame')).toBeInTheDocument();
    expect(await screen.findByText(/Physical Time: Unknown/)).toBeInTheDocument();
    expect(screen.getByRole('link',{name:'Export Swing Package'})).toHaveAttribute('href','/api/v1/necromatcher/swings/hogan-practice/export');
  });
  it('hides old source imagery while the selected frame loads', async () => {
    mocks.frame.mockResolvedValueOnce({capture_id:'capture-v1',frame_index:0,frame_count:3,image_width:320,image_height:240,frame:{pts_ticks:1100,timebase_numerator:1,timebase_denominator:10,physical_time_s:null},observation:{status:'missing',landmarks:{}}}).mockImplementation(() => new Promise(() => {}));
    show('/tools/necromatcher?player=ben-hogan&swing=hogan-practice&capture=capture-v1');
    await screen.findByAltText('Historical Source Frame');
    const slider = screen.getByRole('slider', {name:'Source Frame'});
    slider.focus();
    fireEvent.change(slider, {target: {value: '1'}});
    await waitFor(() => expect(mocks.frame).toHaveBeenCalledWith('capture-v1', 1));
    await waitFor(() => expect(screen.queryByAltText('Historical Source Frame')).not.toBeInTheDocument());
    expect(screen.getByRole('slider', {name:'Source Frame'})).toBeInTheDocument();
  });
  it('saves a new historical player through the library API', async () => {
    mocks.createPlayer.mockResolvedValue({subject_id:'bobby-jones',display_name:'Bobby Jones',metadata:{}});
    const user = userEvent.setup(); show();
    await screen.findByRole('button', {name:'Ben Hogan'});
    await user.type(screen.getByLabelText('Player ID'), 'bobby-jones');
    await user.type(screen.getByLabelText('Player Name'), 'Bobby Jones');
    await user.click(screen.getByRole('button', {name:'Save Player'}));
    await waitFor(() => expect(mocks.createPlayer).toHaveBeenCalledWith('bobby-jones', 'Bobby Jones'));
  });
  it('shows actionable loading failure', async () => {
    mocks.players.mockRejectedValue(new Error('Local library unavailable')); show();
    expect(await screen.findByRole('alert')).toHaveTextContent('Local library unavailable');
  });
  it('hides previous player assets immediately when recalling another URL', async () => {
    const user = userEvent.setup();
    show('/tools/necromatcher?player=ben-hogan&swing=hogan-practice');
    await screen.findByRole('button', {name: /capture-v1/});
    await screen.findByRole('button', {name: 'Hogan Practice'});
    mocks.swings.mockImplementation(() => new Promise(() => {}));
    mocks.assets.mockImplementation(() => new Promise(() => {}));
    await user.click(screen.getByRole('button', {name: 'Recall Tiger URL'}));
    expect(screen.queryByRole('button', {name: /capture-v1/})).not.toBeInTheDocument();
    expect(screen.queryByRole('button', {name: 'Hogan Practice'})).not.toBeInTheDocument();
  });
  it('stops frame loading on failure and retries the same recalled frame', async () => {
    mocks.frame.mockRejectedValueOnce(new Error('Capture archive temporarily unavailable'));
    const user = userEvent.setup();
    show('/tools/necromatcher?player=ben-hogan&swing=hogan-practice&capture=capture-v1');
    expect(await screen.findByRole('alert')).toHaveTextContent('Capture archive temporarily unavailable');
    expect(screen.queryByText('Loading Source Frame…')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', {name: 'Retry Loading'}));
    expect(await screen.findByAltText('Historical Source Frame')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(mocks.frame).toHaveBeenCalledTimes(2);
  });
  it('does not expose another player swing through a mismatched saved URL', async () => {
    show('/tools/necromatcher?player=tiger-woods&swing=hogan-practice&capture=capture-v1');
    await screen.findByRole('button', {name: 'Tiger Practice'});
    expect(mocks.assets).not.toHaveBeenCalled();
    expect(mocks.frame).not.toHaveBeenCalled();
    expect(screen.queryByRole('link', {name: 'Export Swing Package'})).not.toBeInTheDocument();
    expect(screen.queryByRole('button', {name: 'Import Version'})).not.toBeInTheDocument();
  });
});

it('retains export URL recall while changing a selected fit source frame', async () => {
  mocks.assets.mockResolvedValue({assets:[
    {dataset_id:'capture-v1',session_id:'hogan-practice',kind:'image_capture',metadata:{frame_count:3}},
    {dataset_id:'fit-v2',session_id:'hogan-practice',kind:'kinematic_fit',metadata:{capture_id:'capture-v1',qualification:'monocular_research_hypothesis'}},
  ]});
  mocks.videoView.mockResolvedValue({run_id:'saved-export',source_fit_id:'fit-v2',status:'succeeded',acceptance:'rejected',qualification:'monocular_research_hypothesis',blockers:[],message:'Verified export',fraction:null,control_available:false,execution_verified:true,download_available:true});
  mocks.projection.mockImplementation(() => new Promise(() => {}));
  show('/tools/necromatcher?player=ben-hogan&swing=hogan-practice&capture=capture-v1&fit=fit-v2&export_run=saved-export');
  await screen.findByRole('link',{name:'Download Overlay Package'});
  fireEvent.change(screen.getByRole('slider',{name:'Source Frame'}), {target:{value:'1'}});
  expect(screen.getByLabelText('Recalled Selection')).toHaveTextContent('export_run=saved-export');
  expect(mocks.videoView).toHaveBeenCalledWith('saved-export');
});
