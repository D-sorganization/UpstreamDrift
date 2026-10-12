import {useCallback, useState} from 'react';
import {apiFetch} from '@/api/fetch';
import {checkedPrepared, checkedTrajectory, type ResearchPrepared, type ResearchTrajectory} from './researchGolfContracts';
import {cancelResearchPreparation as cancel,useResearchTurnover} from './useResearchTurnover';
const base='/tools/golf-simulator';
const post=<T,>(path:string, body:unknown)=>apiFetch<T>(base+path,{method:'POST',body:JSON.stringify(body)});
export function useResearchGolf(replay:string, run:string) {
  const [destination,setDestination]=useState('local'), [status,setStatus]=useState('DISCONNECTED');
  const [prepared,setPrepared]=useState<ResearchPrepared|null>(null), [token,setToken]=useState<string|null>(null);
  const [trajectory,setTrajectory]=useState<ResearchTrajectory|null>(null), [error,setError]=useState('');
  const [working,setBusy]=useState(false), [session,setSession]=useState('');
  const reset=useCallback(()=>{setPrepared(null);setToken(null);setTrajectory(null);setStatus('DISCONNECTED');setSession('');setBusy(false);setError('');},[]);
  const turnover=useResearchTurnover(`${replay}/${run}`,reset);
  const {epoch,owned,inFlight}=turnover;
  const busy=working || turnover.changing,canConnect=!busy && !turnover.error;
  function clear() {setPrepared(null);setToken(null);setTrajectory(null);}
  async function action(work:(current:()=>boolean)=>Promise<void>) {
    const generation=++epoch.current;setBusy(true);setError('');
    const current=()=>generation===epoch.current;
    try {const operation=work(current);inFlight.current=operation;await operation;} catch(reason) {if(current()) {
      clear();setStatus('FAULT');let message=reason instanceof Error?reason.message:'Research action failed.';
      if(owned.current) {try {await cancel(owned.current);owned.current=null;} catch {message+=' Owned preparation could not be cancelled; reconnect and resolve the server session before continuing.';}}
      if(current()) setError(message);
    }}
    finally {if(current()) setBusy(false);}
  }
  async function replace(next:string) {
    await action(async current=>{clear();setStatus('DISCONNECTED');setSession('');
      if(owned.current) {await cancel(owned.current);owned.current=null;}
      if(current()) setDestination(next);
    });
  }
  async function connect() {
    if(!canConnect) return;
    await action(async current=>{clear();if(owned.current) {await cancel(owned.current);owned.current=null;}
      const response=await post<{state:string;session_id:string}>('/session',{destination_id:destination,session_id:'web-session'});
      if(current()) {if(response.state!=='idle' || !response.session_id) throw new Error('Research requires an idle connected session.');setSession(response.session_id);setStatus('CONNECTED');}
    });
  }
  async function prepare() {
    if(destination!=='local' || !session || busy || !['CONNECTED','ACCEPTED','REJECTED'].includes(status)) return;
    await action(async current=>{clear();const shotId=`research-${crypto.randomUUID()}`;const response=await post<ResearchPrepared>('/shot/prepare-research-impact',{
      replay_id:replay,run_id:run,shot_id:shotId,session_id:session,created_at_utc:new Date().toISOString(),
      aim_context:{source_to_target_rotation:[[1,0,0],[0,1,0],[0,0,1]],revision:1},context_revision:1,
    });
      if(!current()) {if(response.prepared_shot_id) await cancel(response.prepared_shot_id);return;}
      owned.current=response.prepared_shot_id;
      const checked=checkedPrepared(response,replay,run);if(checked.shot_id!==shotId) throw new Error('Foreign research preparation shot.');setPrepared(checked);setStatus('PREPARED');
    });
  }
  async function arm() {if(!prepared || busy) return;await action(async current=>{
    const response=await post<{arm_token:string}>('/shot/arm',{prepared_shot_id:prepared.prepared_shot_id,context_revision:1});
    if(current()) {if(typeof response.arm_token!=='string' || !response.arm_token) throw new Error('Invalid arm token.');setToken(response.arm_token);setStatus('ARMED');}
  });}
  async function submit() {if(!prepared || !token || busy) return;await action(async current=>{
    const response=await post<{state:string;shot_id:string}>('/shot/submit',{prepared_shot_id:prepared.prepared_shot_id,arm_token:token});
    if(!current()) return;
    owned.current=null;setToken(null);
    if(response.shot_id!==prepared.shot_id) throw new Error('Foreign research shot response.');
    if(response.state!=='confirmed_accepted') {setStatus(response.state==='unknown_ambiguous'?'SENT_UNCONFIRMED':'REJECTED');return;}
    const result=await apiFetch<ResearchTrajectory>(`${base}/shot/${encodeURIComponent(response.shot_id)}/local-trajectory`);
    if(current()) {setTrajectory(checkedTrajectory(result,prepared));setStatus('ACCEPTED');}
  });}
  async function stop() {await action(async current=>{if(owned.current) {await cancel(owned.current);owned.current=null;}if(current()) {clear();setStatus('CONNECTED');}});}
  return {destination,status,prepared,trajectory,error:error || turnover.error,busy,canConnect,replace,connect,prepare,arm,submit,stop};
}
