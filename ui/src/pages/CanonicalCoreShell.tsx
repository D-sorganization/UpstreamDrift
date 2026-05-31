import { useMemo } from 'react';

type CanonicalCoreMode = 'estimation' | 'comparison';

interface CanonicalCoreShellPageProps {
  mode: CanonicalCoreMode;
}

const COPY: Record<CanonicalCoreMode, { title: string; body: string; service: string }> = {
  estimation: {
    title: 'Canonical-Core Estimation',
    body: 'CC-19 estimation workspace for canonical-state fit handoff.',
    service: 'canonical-core estimation services',
  },
  comparison: {
    title: 'Canonical-Core Comparison',
    body: 'CC-27 comparison workspace for cross-engine canonical-state review.',
    service: 'canonical-core comparison services',
  },
};

export function CanonicalCoreShellPage({ mode }: CanonicalCoreShellPageProps) {
  const content = COPY[mode];
  const route = useMemo(() => `/tools/canonical-core/${mode}`, [mode]);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <section className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-6 py-8">
        <header className="flex flex-col gap-2 border-b border-slate-800 pb-5">
          <p className="text-sm font-medium uppercase tracking-wide text-cyan-300">
            Canonical Core
          </p>
          <h1 className="text-3xl font-semibold">{content.title}</h1>
          <p className="max-w-3xl text-sm leading-6 text-slate-300">{content.body}</p>
        </header>

        <div className="grid gap-4 md:grid-cols-3">
          <section className="rounded-md border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-sm font-semibold text-slate-100">Shells</h2>
            <p className="mt-3 text-sm leading-6 text-slate-300">
              PyQt6 desktop and React/Tauri web surfaces are both represented by
              the shared launcher manifest.
            </p>
          </section>
          <section className="rounded-md border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-sm font-semibold text-slate-100">Service Boundary</h2>
            <p className="mt-3 text-sm leading-6 text-slate-300">{content.service}</p>
          </section>
          <section className="rounded-md border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-sm font-semibold text-slate-100">Route</h2>
            <p className="mt-3 break-words font-mono text-sm text-cyan-200">{route}</p>
          </section>
        </div>
      </section>
    </main>
  );
}

export default CanonicalCoreShellPage;
