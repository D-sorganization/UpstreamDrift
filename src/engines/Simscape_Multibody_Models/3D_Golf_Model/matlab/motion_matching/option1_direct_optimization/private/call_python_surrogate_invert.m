function phase = call_python_surrogate_invert(target, options)
%CALL_PYTHON_SURROGATE_INVERT  Bridge to Python's fit_swing_via_surrogate.
%
%   PHASE = CALL_PYTHON_SURROGATE_INVERT(TARGET, OPTIONS) calls Python's
%   `src.shared.python.motion_matching.surrogate.fit_swing_via_surrogate`
%   via `pyrun` and returns a struct with the warm-start coefficient
%   vector and metadata. The MATLAB caller is responsible for handing
%   `phase.coefficients` to `fit_swing_fmincon` as `opts.initial_theta`.
%
%   The function delegates to a user-supplied callable when
%   `OPTIONS.surrogate_invert_fn` is set; this is the seam unit tests use
%   to stub out the Python call without touching pyrun.
%
%   Returned PHASE struct fields:
%       coefficients : (d,1) double, the warm-start theta
%       final_loss   : double, the surrogate's reported best loss
%       duration_s   : double
%       n_starts     : double
%       solver       : string == "surrogate"
%
%   GitHub issue: #4000 / #031.
    arguments
        target  (1,1) struct
        options (1,1) struct
    end

    t0 = tic;

    if isfield(options, "surrogate_invert_fn") && ...
            ~isempty(options.surrogate_invert_fn) && ...
            isa(options.surrogate_invert_fn, 'function_handle')
        out = options.surrogate_invert_fn(target, options);
        phase = local_normalize_phase(out, t0);
        return;
    end

    ckpt = local_get(options, "surrogate_checkpoint", "");
    if strlength(string(ckpt)) == 0
        error("call_python_surrogate_invert:noCheckpoint", ...
            "options.surrogate_checkpoint is required when surrogate_invert_fn is not provided");
    end

    % --- pyrun path ------------------------------------------------------
    % We pass the target as plain numeric arrays to avoid struct->dict
    % marshalling issues across the boundary, and let the Python helper
    % rebuild a ClubTarget.
    n_starts = double(local_get(options, "surrogate_n_starts", 8));
    n_iters  = double(local_get(options, "surrogate_n_iters", 200));
    seed     = double(local_get(options, "rng_seed", 0));

    py_code = strjoin([
        "from pathlib import Path"
        "import numpy as np"
        "import torch"
        "from src.shared.python.motion_matching.club_target import ClubTarget, SourceProvenance"
        "from src.shared.python.motion_matching.surrogate import (InvertOptions, SwingSurrogate, fit_swing_via_surrogate)"
        "bundle = torch.load(str(ckpt_path), map_location='cpu')"
        "surrogate = bundle['model'] if isinstance(bundle, dict) and 'model' in bundle else bundle"
        "prov = SourceProvenance(filename='matlab.bin', format='synthetic', subject_id='M', trial_id='0', sha256='0'*64)"
        "target = ClubTarget(time=np.asarray(time), butt=np.asarray(butt), clubhead=np.asarray(clubhead), club_quat=np.asarray(quat), impact_idx=int(impact_idx), source=prov)"
        "opts = InvertOptions(n_starts=int(n_starts), n_iters_per_start=int(n_iters), seed=int(seed))"
        "fit = fit_swing_via_surrogate(target, surrogate, opts)"
        "result = {'coefficients': fit.coefficients.astype('float64'), 'final_loss': float(fit.final_loss)}"
    ], newline);

    py_result = pyrun(py_code, "result", ...
        ckpt_path  = string(ckpt), ...
        time       = target.time, ...
        butt       = target.butt, ...
        clubhead   = target.clubhead, ...
        quat       = target.club_quat, ...
        impact_idx = double(target.impact_idx), ...
        n_starts   = n_starts, ...
        n_iters    = n_iters, ...
        seed       = seed);

    coeffs = double(py_result{"coefficients"});
    out = struct( ...
        'coefficients', coeffs(:), ...
        'final_loss',   double(py_result{"final_loss"}), ...
        'n_starts',     n_starts);
    phase = local_normalize_phase(out, t0);
end

% =====================================================================
function phase = local_normalize_phase(out, t0)
    if ~isstruct(out)
        error("call_python_surrogate_invert:badReturn", ...
            "surrogate_invert_fn must return a struct");
    end
    if ~isfield(out, "coefficients") || isempty(out.coefficients)
        error("call_python_surrogate_invert:badReturn", ...
            "surrogate phase output missing 'coefficients'");
    end
    phase = struct();
    phase.coefficients = double(out.coefficients(:));
    phase.final_loss   = double(local_get(out, "final_loss", NaN));
    phase.n_starts     = double(local_get(out, "n_starts", NaN));
    phase.duration_s   = toc(t0);
    phase.solver       = "surrogate";
end

% =====================================================================
function v = local_get(s, name, default)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default;
    end
end
