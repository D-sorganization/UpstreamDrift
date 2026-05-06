function opts = default_cost_options()
%DEFAULT_COST_OPTIONS Return the default options struct for compute_cost.
%
%   OPTS = DEFAULT_COST_OPTIONS() returns the canonical defaults defined in
%   COST_FUNCTION_SPEC.md § Defaults. Callers override individual fields
%   on the returned struct.
%
%   Fields:
%     w_position          - position-term weight (m^-2)        : 1.0
%     w_orientation       - orientation-term weight (rad^-2)   : 0.1
%     w_anchor_impact     - impact-anchor multiplier on w_pos  : 10.0
%     regularizer         - one of: "total_work" | "peak_power"
%                                 | "torque_l2" | "coeff_l2"
%                                 | "effort_l2" | "smoothness_l2" : "total_work"
%     lambda              - regularizer strength               : 1e-4
%     q_orientation_repr  - "quaternion" | "rotmat"            : "quaternion"
%     time_alignment      - "impact" | "address" | "none"      : "impact"
%     resample_to_hz      - target sample rate                 : 1000
%     tau_reference       - reference torque profile for
%                           "effort_l2" (empty -> zero)        : []
%     regularizer_weights - per-joint weight vector for
%                           "effort_l2"/"smoothness_l2"
%                           (empty -> ones(n_joints,1))        : []
    opts = struct();
    opts.w_position         = 1.0;
    opts.w_orientation      = 0.1;
    opts.w_anchor_impact    = 10.0;
    opts.regularizer        = "total_work";
    opts.lambda             = 1e-4;
    opts.q_orientation_repr = "quaternion";
    opts.time_alignment     = "impact";
    opts.resample_to_hz     = 1000;
    opts.tau_reference      = [];
    opts.regularizer_weights = [];
end
