## Problem

`ui/src/components/simulation/PolynomialGeneratorPanel.tsx` (~lines 103+): the user can Remove every coefficient row and still click "Apply Polynomial", sending `coefficients: []` to the backend. The result is either a backend rejection or an actuator in a bad state, with unclear feedback either way.

## Fix

1. Disable the Remove button when `coefficients.length === 1` (a polynomial needs at least one term), with a `title`/tooltip explaining why.
2. Belt-and-braces: disable "Apply Polynomial" when `coefficients.length === 0` and add a guard in the click handler.
3. While editing, validate coefficient values (reject NaN from empty number inputs) and disable Apply with an inline hint ("fix invalid coefficient") rather than silently sending NaN.
4. Test: removing down to one coefficient disables Remove; Apply disabled with invalid/empty input.

## Acceptance criteria

- It is impossible to submit an empty or NaN-containing coefficient list from the UI; tests pass.

Part of the UI/UX overhaul epic (see tracking issue).
