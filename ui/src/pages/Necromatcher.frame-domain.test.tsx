import { beforeEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { NecromatcherPage } from './Necromatcher';

const mocks = vi.hoisted(() => ({summary: vi.fn(), frame: vi.fn(), projection: vi.fn()}));
vi.mock('@/components/necromatcher/LibraryActions', () => ({LibraryActions: () => null}));
vi.mock('@/components/necromatcher/RefitControls', () => ({RefitControls: () => null}));
vi.mock('@/components/necromatcher/VideoExportControls', () => ({VideoExportControls: () => null}));
vi.mock('@/api/necromatcher', () => ({
  fetchPlayers: async () => ({players: [{subject_id:'tiger',display_name:'Tiger',metadata:{}}]}),
  fetchSwings: async () => ({swings: [{subject_id:'tiger',session_id:'swing',name:'Swing',metadata:{}}]}),
  fetchAssets: async () => ({assets:[
    {session_id:'swing',dataset_id:'capture',kind:'image_capture',metadata:{frame_count:210}},
    {session_id:'swing',dataset_id:'fit',kind:'kinematic_fit',metadata:{capture_id:'capture'}},
  ]}),
  fetchFitSummary: mocks.summary, fetchCaptureFrame: mocks.frame, fetchFitProjection: mocks.projection,
  captureFrameImageUrl: () => '/source.png', swingExportUrl: () => '/export',
}));
beforeEach(() => {
  Object.values(mocks).forEach(mock => mock.mockReset());
  mocks.summary.mockResolvedValue({fit_id:'fit',capture_id:'capture',frame_count:191,frame_indices:Array.from({length:191},(_,i)=>i)});
  mocks.frame.mockImplementation(async (_capture: string,index: number) => ({capture_id:'capture',frame_index:index,frame_count:210,image_width:1280,image_height:720,frame:{pts_ticks:index,timebase_numerator:1,timebase_denominator:30,physical_time_s:null},observation:{status:'missing',landmarks:{}}}));
  mocks.projection.mockImplementation(async (_fit: string,index: number) => ({fit_id:'fit',capture_id:'capture',frame_index:index,points:{},coordinates:'image_pixels'}));
});
function show(frame: number, fit=true) {
  render(<MemoryRouter initialEntries={[`/tools/necromatcher?player=tiger&swing=swing&capture=capture&frame=${frame}${fit?'&fit=fit':''}`]}><NecromatcherPage /></MemoryRouter>);
}
it.each([191,209])('excluded recalled frame %i never reaches fit projection',async index => {
  show(index);
  await waitFor(() => expect(mocks.projection).toHaveBeenCalledWith('fit',0));
  expect(mocks.projection.mock.calls.every(call => call[1]<=190)).toBe(true);
  expect(screen.getByRole('slider',{name:'Source Frame'})).toHaveAttribute('max','190');
});
it('maps sparse slider positions to exact fitted source indices',async () => {
  mocks.summary.mockResolvedValue({fit_id:'fit',capture_id:'capture',frame_count:3,frame_indices:[10,75,190]});
  show(75);
  await waitFor(() => expect(mocks.projection).toHaveBeenCalledWith('fit',75));
  const slider=screen.getByRole('slider',{name:'Source Frame'});
  expect(slider).toHaveAttribute('max','2'); expect(slider).toHaveValue('1');
  fireEvent.change(slider,{target:{value:'2'}});
  await waitFor(() => expect(mocks.projection).toHaveBeenCalledWith('fit',190));
  expect(mocks.projection.mock.calls.map(call=>call[1])).toEqual([75,190]);
});
it.each([{indices:[190,0]},{indices:[]},{indices:[0,0]},{indices:[0,209.5]}])('rejects malformed ordered domain $indices before projection',async ({indices}) => {
  mocks.summary.mockResolvedValue({fit_id:'fit',capture_id:'capture',frame_count:indices.length,frame_indices:indices});
  show(0); await screen.findByRole('alert'); expect(mocks.projection).not.toHaveBeenCalled();
});
it('keeps source-only browsing through frame209',async () => {
  show(209,false);
  await waitFor(() => expect(mocks.frame).toHaveBeenCalledWith('capture',209));
  expect(screen.getByRole('slider',{name:'Source Frame'})).toHaveAttribute('max','209');
  expect(mocks.summary).not.toHaveBeenCalled(); expect(mocks.projection).not.toHaveBeenCalled();
});
it('does not project while fit summary is unavailable',async () => {
  mocks.summary.mockRejectedValue(new Error('Summary unavailable'));
  show(209); await screen.findByRole('alert'); expect(mocks.projection).not.toHaveBeenCalled();
});
