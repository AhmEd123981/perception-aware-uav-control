function [ff, ctrl] = compute_trajectory_feedforward(x_ref, ctrl, params, math)
    %COMPUTE_TRAJECTORY_FEEDFORWARD Stateful acceleration/thrust/attitude feedforward.
    % The acceleration is estimated from reference velocity and filtered, so
    % controllers get nominal trajectory physics without using tracking error.

    if ~isfield(ctrl, 'use_trajectory_feedforward') || ~ctrl.use_trajectory_feedforward
        ff = quadrotor_flatness_from_acceleration(x_ref, zeros(3,1), params, math, trajectory_feedforward_options(ctrl));
        return;
    end

    dt = 0.01;
    if isfield(ctrl, 'dt') && ~isempty(ctrl.dt)
        dt = max(ctrl.dt, 1e-6);
    end

    x_ref = x_ref(:);
    if numel(x_ref) >= 6
        v_ref = x_ref(4:6);
    else
        v_ref = zeros(3, 1);
    end

    if isfield(ctrl, 'ff_prev_vel_ref') && ~isempty(ctrl.ff_prev_vel_ref)
        a_raw = (v_ref - ctrl.ff_prev_vel_ref) / dt;
    else
        a_raw = zeros(3, 1);
    end
    a_raw(~isfinite(a_raw)) = 0;

    alpha = exp(-12 * dt);
    if isfield(ctrl, 'ff_filter_alpha') && ~isempty(ctrl.ff_filter_alpha)
        alpha = ctrl.ff_filter_alpha;
    end
    alpha = max(0, min(0.98, alpha));

    if isfield(ctrl, 'ff_acc_filtered') && ~isempty(ctrl.ff_acc_filtered)
        a_filt = alpha * ctrl.ff_acc_filtered + (1 - alpha) * a_raw;
    else
        a_filt = a_raw;
    end

    options = trajectory_feedforward_options(ctrl);
    ff = quadrotor_flatness_from_acceleration(x_ref, a_filt, params, math, options);

    ctrl.ff_prev_vel_ref = v_ref;
    ctrl.ff_acc_filtered = ff.acc;
end
