import { beforeEach, expect, it, vi } from 'vitest';
import { submitReplayImpact, fetchReplayImpact, cancelReplayImpact, replayImpactDownloadUrl } from './necromatcher';
const request=vi.hoisted(()=>vi.fn());
vi.mock('./fetch',()=>({apiFetch:request,apiFetchForm:vi.fn()}));
vi.mock('./backend',()=>({getApiBase:()=> 'http://backend.test'}));
beforeEach(()=>request.mockReset());
it('sends the exact authored records and budget without paths or inferred fields',async()=>{
  const payload={geometry:{body:'club',local_head_point_m:[0,0,0],local_face_normal:[1,0,0],local_face_up:[0,0,1],mass_kg:0.2,moi_kg_m2:0.001,assumption_description:'Authored'},selection:{recorded_sample_index:3,world_to_flight_rotation:[[1,0,0],[0,1,0],[0,0,1]],world_to_flight_translation_m:[0,0,0],selection_description:'Authored sample'},budget_wall_s:120};
  await submitReplayImpact('replay/a',payload);
  expect(request).toHaveBeenCalledExactlyOnceWith('/api/v1/necromatcher/replays/replay%2Fa/impact-runs',{method:'POST',body:JSON.stringify(payload)});
});
it('uses replay-owned polling, cancellation and bundle download paths',async()=>{
  await fetchReplayImpact('replay/a','run/b');await cancelReplayImpact('replay/a','run/b');
  expect(request.mock.calls).toEqual([
    ['/api/v1/necromatcher/replays/replay%2Fa/impact-runs/run%2Fb'],
    ['/api/v1/necromatcher/replays/replay%2Fa/impact-runs/run%2Fb/cancel',{method:'POST'}],
  ]);
  expect(replayImpactDownloadUrl('replay/a','run/b')).toBe('http://backend.test/api/v1/necromatcher/replays/replay%2Fa/impact-runs/run%2Fb/download');
});
