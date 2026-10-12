import {useEffect,useRef,useState} from 'react';
import {apiFetch} from '@/api/fetch';
export function cancelResearchPreparation(id:string) {
  return apiFetch('/tools/golf-simulator/shot/cancel',{method:'POST',body:JSON.stringify({prepared_shot_id:id})});
}
export function useResearchTurnover(identity:string, reset:()=>void) {
  const epoch=useRef(0),owned=useRef<string|null>(null),inFlight=useRef<Promise<void>|null>(null);
  const closing=useRef<Promise<void>>(Promise.resolve());
  const previous=useRef(identity),[changing,setChanging]=useState(false),[error,setError]=useState('');
  useEffect(()=>{
    if(previous.current===identity) return;
    previous.current=identity;const generation=++epoch.current;
    async function closePrevious() {
      setChanging(true);setError('');reset();
      try {
        const previousClose=closing.current;
        const nextClose=previousClose.then(async()=>{
          await inFlight.current;
          if(owned.current) {await cancelResearchPreparation(owned.current);owned.current=null;}
        });
        closing.current=nextClose;await nextClose;
        if(generation===epoch.current) setChanging(false);
      } catch(reason) {
        if(generation===epoch.current) {setChanging(false);setError(`Previous research operation could not close: ${reason instanceof Error?reason.message:'cancellation failed'}. New connection is blocked.`);}
      }
    }
    void closePrevious();
  },[identity,reset]);
  useEffect(()=>()=>{epoch.current++;if(owned.current) void cancelResearchPreparation(owned.current).catch(()=>{});},[]);
  return {epoch,owned,inFlight,changing,error};
}
