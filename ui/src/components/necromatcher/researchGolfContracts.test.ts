import {expect,it} from 'vitest';
import {checkedPrepared,checkedTrajectory,researchIdentity} from './researchGolfContracts';
const run='a'.repeat(32);
function prepared() {return {prepared_shot_id:'prep',shot_id:'shot',context_revision:1,is_armed:false,created_at_utc:'now',research:{replay_id:'replay',run_id:run,result_sha256:'sha256:'+'1'.repeat(64),receipt_sha256:'sha256:'+'2'.repeat(64),trajectory_sha256:'sha256:'+'3'.repeat(64),recorded_time_s:.25,assumptions:{geometry:'Authored assumption'},qualification:{contact:'unverified',numerical:'unverified',scientific:'unverified'},replay_clock_policy:'authored_simulation_seconds',scientific_qualified:false,physical_source_time_qualified:false}};}
function trajectory() {return {shot_id:'shot',provenance:'local_reference',simulated_at_utc:'now',research:prepared().research,samples:[{time_s:0,position_m:[0,0,0],velocity_mps:[20,0,5]},{time_s:1,position_m:[20,0,4],velocity_mps:[19,0,3]}]};}
it.each(['foreign shot','changed hash','boolean pixel','nonfinite velocity','nonmonotonic clock','empty'])('rejects %s local output',kind=>{
 const value=trajectory();
 if(kind==='foreign shot') value.shot_id='other';
 if(kind==='changed hash') value.research.result_sha256='sha256:'+'4'.repeat(64);
 if(kind==='boolean pixel') (value.samples[0].position_m as unknown[])[0]=true;
 if(kind==='nonfinite velocity') value.samples[0].velocity_mps[0]=Infinity;
 if(kind==='nonmonotonic clock') value.samples[1].time_s=0;
 if(kind==='empty') value.samples=[];
 expect(()=>checkedTrajectory(value,checkedPrepared(prepared(),'replay',run))).toThrow();
});
it('retains exact sample arrays and detached assumptions across reordered JSON keys',()=>{
 const original=prepared(), admitted=checkedPrepared(original,'replay',run), value=trajectory();
 value.research=Object.fromEntries(Object.entries(value.research).reverse()) as typeof value.research;
 const output=checkedTrajectory(value,admitted);
 expect(output.samples).toEqual(value.samples);
 original.research.assumptions.geometry='Changed';value.samples[0].position_m[0]=99;
 expect(admitted.research.assumptions.geometry).toBe('Authored assumption');expect(output.samples[0].position_m[0]).toBe(0);
});
it('refuses missing/duplicate/path-like query identities',()=>{
 for(const query of ['?replay=replay',`?replay=replay&replay=other&impactRun=${run}`,`?replay=../foreign&impactRun=${run}`]) expect(researchIdentity(query)).toBeNull();
 expect(researchIdentity(`?replay=saved.replay-v1&impactRun=${run}`)).toEqual({replay:'saved.replay-v1',run});
});
