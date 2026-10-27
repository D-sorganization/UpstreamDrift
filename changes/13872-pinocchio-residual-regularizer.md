### Fixed

- Append regulariser residual to Pinocchio `fit_swing` residual vector matching docstring specification, and enforce torque coefficient box bounds via `least_squares(method="trf", bounds=(lb, ub))` (`#13872`).
