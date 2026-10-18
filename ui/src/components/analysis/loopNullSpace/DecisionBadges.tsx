import type { DecisionView } from '@/api/loopNullSpace';

/**
 * A decision's label, badges and chips as visible text (never colour alone):
 * the label and badges say how the quantity is to be read, the chips name the
 * hypotheses and conditioning it rests on.
 */
export function DecisionBadges({ decision, testId }: { decision: DecisionView; testId: string }) {
  const chips: Array<[string, string[]]> = [
    ['Badge', decision.badges],
    ['Hypothesis', decision.hypothesis_chips],
    ['Conditioning', decision.conditioning_chips],
  ];
  return (
    <div data-testid={testId} className="flex flex-wrap gap-1 text-[10px] text-gray-200">
      {decision.label && (
        <span className="rounded border border-gray-500 px-1 font-semibold">
          Label: {decision.label}
        </span>
      )}
      {chips.flatMap(([kind, items]) =>
        items.map((item) => (
          <span key={`${kind}:${item}`} className="rounded border border-gray-600 px-1">
            {kind}: {item}
          </span>
        )),
      )}
      {decision.preset && (
        <span className="rounded border border-gray-600 px-1">Preset: {decision.preset}</span>
      )}
    </div>
  );
}
