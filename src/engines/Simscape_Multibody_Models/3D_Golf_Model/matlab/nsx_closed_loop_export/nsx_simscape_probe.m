function dyn = nsx_simscape_probe(mdl, q, v, closureRows)
%NSX_SIMSCAPE_PROBE  M, h, J_c and Jdot_v of an open-tree Simscape model at (q, v).
%
%   DYN = NSX_SIMSCAPE_PROBE(MDL, Q, V, ROWS) probes the model MDL (a loaded or
%   loadable open-tree Simscape Multibody model that satisfies the probe
%   contract of NSX_BUILD_PLANAR_RIG) by forward dynamics at t = 0 and returns
%       dyn.M       n x n   mass matrix
%       dyn.h       n x 1   velocity-product plus gravity vector
%       dyn.J_c     m x n   closure Jacobian d(Phi)/dq, rows ROWS of [x y z]
%       dyn.Jdot_v  m x 1   closure drift (d/dt J_c) v
%       dyn.phi     m x 1   closure offset at q (a closed configuration has 0)
%       dyn.grip    3 x 1   World position of the grip point
%       dyn.jv_residual     max |J_c v - phidot| (self-consistency check)
%       dyn.n_sims          number of Simscape runs
%
%   Method (no model internals, no finite differences):
%     * acc0 = a(tau = 0) = -M^-1 h at (q, v);  a(tau = e_j) - acc0 = M^-1 e_j,
%       so M = inv([M^-1 columns]) and h = -M acc0.
%     * J_c e_j = phidot at q with velocity target e_j (the sensor velocity is
%       linear in the joint velocities).
%     * Jdot_v = phiddot(tau = 0) - J_c acc0, because phiddot = J_c a + Jdot_v.
%   Joint positions and velocities are set as high-priority targets so the t = 0
%   state is exactly (q, v).

    arguments
        mdl (1,:) char
        q (:,1) double {mustBeFinite}
        v (:,1) double {mustBeFinite}
        closureRows (1,:) double {mustBeInteger, mustBePositive}
    end
    n = numel(q);
    assert(numel(v) == n, 'nsx_simscape_probe:size', 'q and v must have equal length');
    run0 = run_case(mdl, q, v, zeros(n, 1), n);
    minv = zeros(n);
    for j = 1:n
        e = zeros(n, 1); e(j) = 1;
        rj = run_case(mdl, q, v, e, n);
        minv(:, j) = rj.acc - run0.acc;
    end
    mass = inv(minv);
    mass = 0.5 * (mass + mass.');
    jac = zeros(numel(closureRows), n);
    for j = 1:n
        e = zeros(n, 1); e(j) = 1;
        rj = run_case(mdl, q, e, zeros(n, 1), n);
        jac(:, j) = rj.phidot(closureRows);
    end
    dyn.M = mass;
    dyn.h = -mass * run0.acc;
    dyn.J_c = jac;
    dyn.Jdot_v = run0.phiddot(closureRows) - jac * run0.acc;
    dyn.phi = run0.phi(closureRows);
    dyn.grip = run0.grip;
    dyn.jv_residual = max(abs(jac * v - run0.phidot(closureRows)));
    dyn.n_sims = 2 * n + 1;
end

function r = run_case(mdl, q, v, tau, n)
    si = Simulink.SimulationInput(mdl);
    si = si.setVariable('nsx_q', q);
    si = si.setVariable('nsx_v', v);
    si = si.setVariable('nsx_tau', tau);
    out = sim(si);
    if ~isempty(out.ErrorMessage)
        error('nsx_simscape_probe:simFailed', 'Simscape run failed: %s', out.ErrorMessage);
    end
    y = out.yout;
    if isa(y, 'Simulink.SimulationData.Dataset')
        vals = cell(1, y.numElements);
        for k = 1:y.numElements
            ts = y{k}.Values;
            assert(numel(ts.Time) == 1, 'nsx_simscape_probe:samples', ...
                'expected a single t = 0 sample, got %d', numel(ts.Time));
            vals{k} = double(ts.Data(:));
        end
    else
        vals = num2cell(y(1, :));
    end
    assert(numel(vals) == n + 4, 'nsx_simscape_probe:ports', ...
        'expected %d outports, got %d', n + 4, numel(vals));
    r.acc = cellfun(@(c) c(1), vals(1:n)).';
    r.phi = vals{n + 1}(:);
    r.phidot = vals{n + 2}(:);
    r.phiddot = vals{n + 3}(:);
    r.grip = vals{n + 4}(:);
end
