function starts = sample_starting_points(n_starts, lb, ub, strategy, seed)
%SAMPLE_STARTING_POINTS  Build N starting points spanning [lb, ub].
%
%   STARTS = SAMPLE_STARTING_POINTS(N, LB, UB, STRATEGY, SEED) returns a
%   d-by-N matrix of starting points where d = numel(LB) = numel(UB), each
%   column strictly within [LB, UB]. STRATEGY is one of:
%
%       "random"          - uniform iid via rng(seed)
%       "latin_hypercube" - lhsdesign (Statistics & ML Toolbox)
%       "sobol"           - sobolset(d).Skip(1).Leap(0), mapped to [lb,ub]
%
%   Sobol is the project default per issue #3994.
%
%   Preconditions:
%     - n_starts >= 1, integer.
%     - numel(lb) == numel(ub) >= 1, all(lb < ub).
%     - strategy is one of the above.
%
%   Postconditions:
%     - size(starts) == [numel(lb), n_starts]
%     - all starts strictly within [lb, ub]
%     - all columns are pairwise distinct
%
%   GitHub issue: #025 / #3994.
    arguments
        n_starts (1,1) double {mustBePositive, mustBeInteger}
        lb       (:,1) double
        ub       (:,1) double
        strategy (1,1) string = "sobol"
        seed     (1,1) double {mustBeNonnegative, mustBeInteger} = 42
    end

    assert(numel(lb) == numel(ub), ...
        "sample_starting_points:dimMismatch", ...
        "lb and ub must be the same length");
    assert(all(lb < ub), ...
        "sample_starting_points:degenerate", ...
        "lb must be strictly less than ub elementwise");

    d = numel(lb);
    span = ub - lb;

    switch lower(strategy)
        case "random"
            rng(double(seed), "twister");
            U = rand(d, n_starts);

        case "latin_hypercube"
            if exist('lhsdesign', 'file') ~= 2
                warning("sample_starting_points:noLhs", ...
                    "lhsdesign not available; falling back to Sobol");
                U = local_sobol(d, n_starts);
            else
                rng(double(seed), "twister");
                U = lhsdesign(n_starts, d, 'criterion', 'maximin')';
            end

        case "sobol"
            U = local_sobol(d, n_starts);

        otherwise
            error("sample_starting_points:badStrategy", ...
                "Unknown strategy ""%s""", strategy);
    end

    % Map [0,1] -> (lb, ub). Clamp to a tiny interior margin so we are
    % strictly inside the bounds (fmincon tolerates equality but the
    % postcondition asserts strict containment).
    margin = 1e-9;
    U = min(max(U, margin), 1 - margin);
    starts = lb + span .* U;

    % Postconditions
    assert(isequal(size(starts), [d, n_starts]), ...
        "sample_starting_points:badShape", ...
        "Postcondition: starts must be d-by-n_starts");
    assert(all(starts(:) > lb - eps) && all(starts(:) < ub + eps), ...
        "sample_starting_points:outOfBounds", ...
        "Postcondition: all starts must lie within [lb, ub]");
    if n_starts > 1
        assert(size(unique(starts.', 'rows'), 1) == n_starts, ...
            "sample_starting_points:duplicates", ...
            "Postcondition: starting points must be pairwise distinct");
    end
end

function U = local_sobol(d, n)
%LOCAL_SOBOL  Sobol points in [0,1)^d, returned as d-by-n.
    if exist('sobolset', 'file') ~= 2
        % Fallback to a deterministic Halton-ish sequence so the function
        % keeps working without the Statistics & ML toolbox in CI.
        warning("sample_starting_points:noSobol", ...
            "sobolset not available; using Halton fallback");
        U = local_halton(d, n);
        return;
    end
    s = sobolset(d, 'Skip', 1, 'Leap', 0);
    pts = net(s, n);
    U = pts.';   % d-by-n
end

function U = local_halton(d, n)
%LOCAL_HALTON  Minimal Halton sequence using the first d primes.
    primes_list = local_first_primes(d);
    U = zeros(d, n);
    for k = 1:d
        b = primes_list(k);
        for i = 1:n
            U(k, i) = local_radical_inverse(i, b);
        end
    end
end

function r = local_radical_inverse(i, b)
    r = 0;
    f = 1 / b;
    while i > 0
        r = r + f * mod(i, b);
        i = floor(i / b);
        f = f / b;
    end
end

function p = local_first_primes(n)
    p = zeros(n, 1);
    count = 0;
    candidate = 2;
    while count < n
        if isprime(candidate)
            count = count + 1;
            p(count) = candidate;
        end
        candidate = candidate + 1;
    end
end
