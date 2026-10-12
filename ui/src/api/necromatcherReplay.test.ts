import { beforeEach, expect, it, vi } from 'vitest';
import { fetchReplaySummary, importAsset, replayDataUrl } from './necromatcher';
const request=vi.hoisted(() => vi.fn());
vi.mock('./fetch',()=>({apiFetch:request,apiFetchForm:vi.fn()}));
vi.mock('./backend',()=>({getApiBase:()=> 'http://backend.test'}));
beforeEach(()=>request.mockReset());
it('encodes replay identities through the existing verified summary and data routes',async()=>{
  await fetchReplaySummary('replay/a');
  expect(request).toHaveBeenCalledExactlyOnceWith('/api/v1/necromatcher/replays/replay%2Fa');
  expect(replayDataUrl('replay/a')).toBe('http://backend.test/api/v1/necromatcher/replays/replay%2Fa/data');
});
it('imports a replay using only the authored asset identity and server-local source',async()=>{
  await importAsset('swing/a','replays',{id:'replay',source_path:'C:/replay.h5'});
  expect(request).toHaveBeenCalledExactlyOnceWith('/api/v1/necromatcher/swings/swing%2Fa/replays',{
    method:'POST',body:JSON.stringify({id:'replay',source_path:'C:/replay.h5'}),timeoutMs:300_000,
  });
});
