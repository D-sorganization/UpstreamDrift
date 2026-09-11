# Simscape Matching Resumption Plan

## Assessment

The execution pathway is established: R2025b on DeskComputer, native initialization, one forward wrapper, continuous polynomial efforts, identity-checked prefix transfer, checkpoints, cold replay and all-27 actuator audits. A lower-cost coding agent can operate this pathway with bounded instructions and explicit escalation gates. This is an engineering assessment of the task, not a benchmark of any particular model.

The complete scientific solution is not established. The longest verified prefix covers 0.6 of 1.813889 seconds (about one third), at 9.062024 mm RMS. Both latest optimizations exhausted their budgets without convergence. The full capture has missing observations; fixed rigid marker attachments and connected-model constraints leave residuals that torque optimization alone may not remove. Sampled calibrated pose RMS of 26.824744 mm is a local kinematic diagnostic, not a certified global error floor. Geometry and attachment choices remain provisional.

## First Bounded Execution Assignment

1. Read AGENT_HANDOFF.md and RESUME_AND_STORAGE.md. Confirm DeskComputer connectivity, explicit R2025b, free scratch space, no competing fit in the intended runtime, and unchanged source revision. Keep the saved 0.6 s run immutable.
2. Use original-state runtime 0715f95f3 and its existing runner. Create a new uniquely named run for 0.7 s, cubic basis, finite-difference step 0.00001 and max-nfev 10. Transfer from prefix-600ms-cubic-refine-02/first_prefix_fit.json with the original qualified seed. These are proposed starting settings, not claimed optimal settings. Use the saved launch script/arguments as the template; preserve the transfer report and exact source identity.
3. Keep geometry, offsets, q0 and qd0 fixed. Every simulation begins at global t0; extending the fitted interval never means restarting dynamics from a measured intermediate pose. Save every checkpoint and actual nonblank process exit code.
4. On completion, run the existing independent R2025b cold replay and audit all 27 actuator channels. Retain the existing initialization/projection/effort tolerances. If any invariant fails, preserve the failed evidence and stop this assignment for diagnosis.
5. Produce the existing diagnostic chart plus numerical comparison over the shared 0-0.6 s interval and the new 0.6-0.7 s interval separately. Report RMS, p95, maximum, per-marker residuals, effort ranges/bound saturation, convergence status and native evaluation count. Overall RMS alone cannot show whether extending the interval sacrificed the earlier match.
6. Save immutable raw archives on both machines, verify ZIP CRC and cross-machine SHA256, commit reports/receipts and update handoff before any subsequent stage. Finish this bounded assignment with a recommendation based on the actual results.

## Decisions After the First Assignment

A reviewer with stronger scientific reasoning should assess native stability, old-prefix degradation, localized residuals, torque saturation and improvement per evaluation. Select the next duration and effort degree from evidence rather than automatically running to the end. If cubic profiles lack flexibility, the existing tested degree elevation supports degrees through six while preserving the transferred curve; using a higher degree is a new experiment, not an acceptance shortcut.

If the fit plateaus, distinguish insufficient optimization budget, finite-difference sensitivity, polynomial flexibility, initialization, attachment mismatch and model geometry. Preserve the best prior candidate. Do not loosen tolerances or call a budget-limited result converged. Two successive refinements with little improvement should trigger review instead of repeated blind computation; the reviewer must define any quantitative plateau threshold before automating that decision.

Continue to 0.8 s and onward only after reviewing the previous stage. Reduce the extension near difficult motion rather than forcing a fixed increment. Missing-marker selection becomes material from 1.233333 s for this capture. Include the final frame at 1.813889 s; the existing sampled pose diagnostic ends at 1.8 s and does not cover that final tail.

Keep the calibrated-state experiment separate. Its 0.1 s residual is 12.191880 mm because its fixed multiframe offsets trade initial coincidence for broader kinematic fit. Compare both identities at the same duration and with the same valid observations before selecting one. Changing geometry, offsets or initial state requires fresh native qualification and invalidates an old-identity torque transfer.

## Division of Work

| Work                                                                                  | Suitable Assignment                              |
| ------------------------------------------------------------------------------------- | ------------------------------------------------ |
| Launch prescribed runs, monitor, collect exits and checkpoints                        | Lower-Cost Coding Agent                          |
| Cold replay, actuator checks, plots, hashes, handoff updates                          | Lower-Cost Coding Agent Using Existing Scripts   |
| Fix routine launcher failures without changing physics                                | Coding Agent; Escalate Uncertain Failures        |
| Choose geometry/attachments, alter solver or objective, diagnose structural residuals | Strong Scientific Reviewer                       |
| Change tolerances, define final acceptance, assess physiological realism              | Strong Scientific Reviewer and User Requirements |

No new agent or task is launched by this plan. Switching to a cheaper agent reduces orchestration cost; MATLAB simulation time remains. The last refinement used 877 native evaluations and about 65 minutes for fitting on DeskComputer, plus cold validation. This is a historical measurement, not a forecast for longer intervals. Avoid unnecessary reasoning while native processes run; use durable checkpoints and bounded monitoring.

## Completion Evidence

A representative full-swing deliverable requires native forward integration across all capture frames, continuous effort profiles, a declared fixed geometry/attachment identity, independent replay and all-27 audit, an overlay animation with valid C3D markers, residual-by-time/body/marker plots, and executable reproduction instructions for a second target file. Report achieved accuracy and remaining structural limitations honestly; a universal 5 mm target has not been established as feasible for this model. Full-swing optimization, sensitivity/robustness and effort/physiological qualification still need evidence. New code continues to require TDD, explicit contracts and reuse of existing helpers.
