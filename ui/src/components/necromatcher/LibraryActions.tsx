import { useState, type FormEvent } from 'react';
import { createPlayer, createSwing, importAsset } from '@/api/necromatcher';

const field = 'block w-full mt-1 rounded border border-gray-600 bg-gray-900 p-2 text-gray-100';
const action = 'rounded bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-500 disabled:opacity-50';

function ActionForm({ title, fields, onSave }: {
  title: string; fields: Array<{name: string; label: string}>;
  onSave: (values: Record<string, string>) => Promise<unknown>;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const values = Object.fromEntries(Array.from(new FormData(form).entries()).map(([key, value]) => [key, String(value).trim()]));
    setBusy(true); setError('');
    try { await onSave(values); form.reset(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'The library action failed.'); }
    finally { setBusy(false); }
  }
  return <form onSubmit={submit} className="space-y-3 border-t border-gray-700 pt-4">
    {fields.map((item) => <label className="block text-sm text-gray-300" key={item.name}>{item.label}<input name={item.name} className={field} required disabled={busy} /></label>)}
    {error && <p role="alert" className="text-sm text-red-300">{error}</p>}
    <button className={action} disabled={busy} type="submit">{busy ? 'Saving…' : title}</button>
  </form>;
}

export function LibraryActions({ player, swing, onChanged }: {player: string; swing: string; onChanged: () => void}) {
  const [kind, setKind] = useState<'captures' | 'models' | 'profiles'>('captures');
  const [engine, setEngine] = useState('mujoco');
  const [dofs, setDofs] = useState('');
  async function save(actionPromise: Promise<unknown>) { await actionPromise; onChanged(); }
  return <div className="space-y-4">
    <h2 className="text-lg font-semibold text-white">Library Actions</h2>
    <ActionForm title="Save Player" fields={[{name:'id',label:'Player ID'},{name:'name',label:'Player Name'}]} onSave={(values) => save(createPlayer(values.id, values.name))} />
    {player && <ActionForm title="Save Swing" fields={[{name:'id',label:'Swing ID'},{name:'name',label:'Swing Name'}]} onSave={(values) => save(createSwing(values.id, player, values.name))} />}
    {swing && <div className="space-y-3">
      <label className="block text-sm text-gray-300">Import Type<select className={field} value={kind} onChange={(event) => setKind(event.target.value as typeof kind)}>
        <option value="captures">Capture Folder</option><option value="models">Native Model</option><option value="profiles">Torque Profile</option>
      </select></label>
      {kind === 'models' && <><label className="block text-sm text-gray-300">Model Engine<select className={field} value={engine} onChange={(event) => setEngine(event.target.value)}>{['mujoco','drake','pinocchio','opensim','simscape'].map((name) => <option key={name}>{name}</option>)}</select></label>
        <label className="block text-sm text-gray-300">Ordered Joint Names<input className={field} value={dofs} onChange={(event) => setDofs(event.target.value)} placeholder="hip, knee, shoulder" /></label></>}
      <p className="text-xs text-gray-400">Use a local capture folder, model file or profile JSON. Models are saved as candidates until native replay is verified.</p>
      <ActionForm title="Import Version" fields={[{name:'id',label:'Version ID'},{name:'source_path',label:'Source Path'}]} onSave={(values) => save(importAsset(swing, kind, {id:values.id,source_path:values.source_path,...(kind === 'models' ? {engine,dofs:dofs.split(',').map((x) => x.trim()).filter(Boolean)} : {})}))} />
    </div>}
  </div>;
}
