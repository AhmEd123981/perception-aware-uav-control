function [u, ctrl] = controller_LQR_compute(uav, x_ref, ctrl, params, math)
    %% LQR Controller Computation
    % Full-state feedback:  u = u_ff + K * e
    % with Z-axis integral action for altitude tracking.

    ctrl.last_failure = false;
    try
        %% Current state
        x_pos  = uav.x;
        v      = uav.v;
        R_mat  = uav.R;
        omega  = uav.W;
        euler  = math.dcm_to_euler(R_mat);

        x_current = [x_pos; v; euler; omega];

        %% Reference state
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
            q_ref     = x_ref(7:10);
            R_ref     = math.quat_to_dcm(q_ref);
            euler_ref_full = math.dcm_to_euler(R_ref);
            euler_ref = [0; 0; euler_ref_full(3)];
        end

        x_reference = [x_ref_pos; v_ref; euler_ref; omega_ref];

        %% State error
        e    = x_reference - x_current;
        e(9) = atan2(sin(e(9)), cos(e(9)));   % wrap yaw

        %% Integral action (Z-axis only)
        if ctrl.use_integral
            ctrl.integral_pos(3) = ctrl.integral_pos(3) + e(3) * ctrl.dt;
            ctrl.integral_pos(3) = max(-ctrl.integral_limit, ...
                                   min( ctrl.integral_limit, ctrl.integral_pos(3)));
            z_integral_thrust = params.m * ctrl.Ki_pos(3) * ctrl.integral_pos(3);
        else
            z_integral_thrust = 0;
        end

        %% LQR control law
        u_feedback    = ctrl.K * e;
        u_feedforward = [thrust_ff; 0; 0; 0];

        u = u_feedforward + u_feedback;
        u(1) = u(1) + z_integral_thrust;

        %% Constraints
        u(1)   = max(0.1*params.m*params.g, min(2.0*params.m*params.g, u(1)));
        if isfield(params, 'torque_max')
            u(2:4) = max(-params.torque_max, min(params.torque_max, u(2:4)));
        end

        %% Validate
        if any(~isfinite(u))
            ctrl.last_failure = true;
            warning('LQR output non-finite');
            u = [params.m * params.g; 0; 0; 0];
            if ctrl.use_integral
                ctrl.integral_pos = zeros(3,1);
            end
        end

    catch ME
        ctrl.last_failure = true;
        warning('LQR controller computation failed: %s', ME.message);
        u = [params.m * params.g; 0; 0; 0];
    end
end
