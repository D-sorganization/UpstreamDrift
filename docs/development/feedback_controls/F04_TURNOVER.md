# F04a Coupled Loop Tuning Turnover (#11859; Parent #11788)

This branch adds `control_loop_tuning.py` as a bounded outer tuner for the
existing F02 controller. It uses block-coordinate initialization, a bounded
joint refinement, checkpointed accept/reject decisions, heldout full/reduced
reports, scaled phase/group Jacobian and objective cross-Hessian diagnostics,
rank flags, descriptive phase covariance, and frozen-vs-refit perturbation
comparison. Every evaluator rollout is a caller-supplied forward-dynamics run;
the unit fixture actually drives F02 against an independently stepped
mass-coupled two-coordinate plant with exact simulated-state feedback.

The negative tests were written first. They exposed the missing module, then
showed that a loss-sensitivity Gram matrix was not the objective Hessian. The
implementation now uses finite differences of the frozen training objective;
analytic coupled and uncoupled tests verify the off-diagonal value. Further
negatives cover cross-group regression, solver nonconvergence, constraint
violation, phase confounding, parameter permutation, unbounded diagnostics,
missing holdout and nonfinite data.

The code has not run against a native full-body engine or private capture.
Native F06 torque replay, independent observation scoring, contact/force
evidence, model identifiability and multiple independent heldout swings are
needed before F04 can make a scientific or performance claim. F04 remains open
for that integration. Reproduce with
`python -m pytest tests/unit/motion_matching/test_control_loop_tuning.py -q --no-cov`.
