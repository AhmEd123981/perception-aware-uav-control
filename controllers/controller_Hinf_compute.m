function [u, ctrl] = controller_Hinf_compute(uav, x_ref, ctrl, params, math)
%% H-INFINITY CONTROLLER COMPUTATION
% Full-state feedback:  u = u_ff + K * e
% with disturbance observer and Z-axis integral action.

    ctrl.last_failure = false;
    try
        %% Extract current state
        x_pos = uav.x;
        v     = uav.v;
        R_mat = uav.R;
        omega = uav.W;
        euler = math.dcm_to_euler(R_mat);

        %% Extract reference
        x_ref_pos = x_ref(1:3);
        v_ref     = x_ref(4:6);
        omega_ref = x_ref(11:13);

        % Trajectory feedforward
        thrust_ff = params.m * params.g;
        if isfield(ctrl, 'use_trajectory_feedforward') && ctrl.use_trajectory_feedforward
            [ff, ctrl] = compute_trajectory_feedforward(x_ref, ctrl, params, math);
            euler_ref = ff.euler;
            thrust_ff = ff.thrust;
        else
            q_ref      = x_ref(7:10);
            R_ref      = math.quat_to_dcm(q_ref);
            euler_ref_full = math.dcm_to_euler(R_ref);
            euler_ref  = [0; 0; euler_ref_full(3)];
        end

        %% Build full 12-state error
        x_current   = [x_pos; v; euler; omega];
        x_reference = [x_ref_pos; v_ref; euler_ref; omega_ref];
        e           = x_reference - x_current;
        e(9)        = atan2(sin(e(9)), cos(e(9)));

        %% Integral action (Z-axis only)
        if ctrl.use_integral
            ctrl.integral_pos(3) = ctrl.integral_pos(3) + e(3) * ctrl.dt;
            ctrl.integral_pos(3) = max(-ctrl.integral_limit, ...
                                   min( ctrl.integral_limit, ctrl.integral_pos(3)));
            z_integral_thrust = params.m * ctrl.Ki_pos(3) * ctrl.integral_pos(3);
        else
            z_integral_thrust = 0;
        end

        %% Disturbance observer
        if ctrl.use_observer
            if ~isfield(ctrl, 'observer_initialized') || ~ctrl.observer_initialized
                ctrl.v_prev = v;
                ctrl.R_prev = R_mat;
                ctrl.u_prev = [thrust_ff; 0; 0; 0];
                ctrl.observer_initialized = true;
            else
                thrust_prev = ctrl.u_prev(1);
                F_model     = params.m * [0; 0; -params.g] + thrust_prev * ctrl.R_prev * [0;0;1];
                a_model     = F_model / params.m;

                a_meas = (v - ctrl.v_prev) / ctrl.dt;

                d_meas       = params.m * (a_meas - a_model);
                alpha_obs    = ctrl.L_obs * ctrl.dt / (1 + ctrl.L_obs * ctrl.dt);
                ctrl.d_hat   = (1 - alpha_obs) * ctrl.d_hat + alpha_obs * d_meas;

                max_dist = 0.4 * params.m * params.g;
                if norm(ctrl.d_hat) > max_dist
                    ctrl.d_hat = ctrl.d_hat * (max_dist / norm(ctrl.d_hat));
                end

                e(4:6) = e(4:6) - ctrl.d_hat / params.m * 0.08;
            end
        end

        % Save state for next step
        ctrl.v_prev = v;
        ctrl.R_prev = R_mat;

        %% H-inf full-state feedback
        u_feedback    = ctrl.K * e;
        u_feedforward = [thrust_ff; 0; 0; 0];

        u = u_feedforward + u_feedback;
        u(1) = u(1) + z_integral_thrust;

        %% Constraints
        u(1)   = max(0.2*params.m*params.g, min(2.0*params.m*params.g, u(1)));
        u(2:4) = max(-params.torque_max, min(params.torque_max, u(2:4)));

        % Save for observer
        ctrl.u_prev = u;

        if any(~isfinite(u))
            ctrl.last_failure = true;
            u = [params.m*params.g; 0; 0; 0];
            ctrl.d_hat        = zeros(3,1);
            ctrl.integral_pos = zeros(3,1);
        end

    catch ME
        ctrl.last_failure = true;
        warning('H-inf compute failed: %s', ME.message);
        u = [params.m*params.g; 0; 0; 0];
        ctrl.d_hat        = zeros(3,1);
        ctrl.integral_pos = zeros(3,1);
    end
end
