## Problem

`ui/src/components/ux/HelpfulField.tsx` (~line 27): when a numeric value falls outside the field's `validRange`, the component fires the `onViolation` callback but renders **no visible error** — the user gets no feedback unless the parent happens to surface it (most callers don't). Out-of-range values look accepted (WCAG 3.3.1 Error Identification).

## Fix

1. Track violation state inside HelpfulField and render it:
   ```tsx
   const [violation, setViolation] = useState<string | null>(null);
   // in handleChange, when parsed value is out of range:
   setViolation(
     `Value must be between ${meta.validRange[0]} and ${meta.validRange[1]} ${
       meta.unit ?? ""
     }`,
   );
   // render below the input:
   {
     violation && (
       <p className="text-xs text-red-400" role="alert">
         {violation}
       </p>
     );
   }
   ```
   Clear it when the value returns to range. Keep firing `onViolation` for parents that aggregate.
2. Add `aria-invalid={!!violation}` and `aria-describedby` pointing at the error paragraph; give the input a red border while invalid (`border-red-500`).
3. Tests in `HelpfulField.test.tsx`: out-of-range input shows the message with `role="alert"` and `aria-invalid`; back-in-range clears it.

## Acceptance criteria

- Out-of-range entry is visibly and programmatically flagged at the field; tests pass.

Part of the UI/UX overhaul epic (see tracking issue).
