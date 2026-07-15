function [u, ctrl] = controller_PID_compute(uav, x_ref, ctrl, params, math)
    %% Enhanced PID Controller Computation
    % Improved implementation with robust error handling
    
    ctrl.last_failure = false;
    try
        % Extract current state
        x_pos = uav.x;
        v = uav.v;
        R = uav.R;
        omega = uav.W;
        
        % Extract reference
        x_ref_pos = x_ref(1:3);
        v_ref = x_ref(4:6);
        % q_ref = x_ref(7:10);  % مش محتاجين ده
        omega_ref = x_ref(11:13);
        
        % محذوف: Convert reference quaternion to rotation matrix
        % R_ref = quat_to_rotation_matrix(q_ref);  % مش محتاجين ده
        
        % Position control
        e_pos = x_ref_pos - x_pos;
        e_vel = v_ref - v;
        
        % Update position integral with anti-windup
        dt = ctrl.dt;
        ctrl.integral_pos = ctrl.integral_pos + e_pos * dt;
        for i = 1:3
            if abs(ctrl.integral_pos(i)) > ctrl.integral_limit_pos(i)
                ctrl.integral_pos(i) = sign(ctrl.integral_pos(i)) * ctrl.integral_limit_pos(i);
            end
        end
        
        % Trajectory acceleration feedforward plus measured velocity-error damping.
        % Avoid differentiating noisy position error at 100 Hz.
        if isfield(ctrl, 'use_trajectory_feedforward') && ctrl.use_trajectory_feedforward
            [ff, ctrl] = compute_trajectory_feedforward(x_ref, ctrl, params, math);
            a_ff = ff.acc;
        else
            a_ff = zeros(3, 1);
        end
        if ~isfield(ctrl, 'filtered_vel_error') || isempty(ctrl.filtered_vel_error)
            ctrl.filtered_vel_error = zeros(3, 1);
        end
        ctrl.filtered_vel_error = ctrl.alpha * ctrl.filtered_vel_error + (1 - ctrl.alpha) * e_vel;
        
        % Position control law in force units.
        F_des = params.m * (params.g * [0; 0; 1] + a_ff) + ...
                ctrl.kp_pos .* e_pos + ...
                ctrl.ki_pos .* ctrl.integral_pos + ...
                ctrl.kd_pos .* ctrl.filtered_vel_error;
        
        % Desired thrust
        thrust_des = norm(F_des);
        thrust_min = 0.2 * params.m * params.g;
        thrust_max = 1.8 * params.m * params.g;
        thrust_cmd = max(thrust_min, min(thrust_max, thrust_des));
        
        % Desired thrust direction
        if thrust_des > 1e-6
            b3_des = F_des / thrust_des;
            b3_des = b3_des / norm(b3_des);
        else
            b3_des = [0; 0; 1];
        end
        
        % Current thrust direction
        b3_current = R(:, 3);
        
        % Attitude error
        e_att_vec = cross(b3_current, b3_des);
        if norm(e_att_vec) > 1e-6
            rotation_axis = e_att_vec / norm(e_att_vec);
            rotation_angle = asin(min(1.0, norm(e_att_vec)));
            e_att = rotation_angle * rotation_axis;
        else
            e_att = zeros(3, 1);
        end
        
        % Angular velocity error
        e_omega = omega_ref - omega;
        
        % Update attitude integral with anti-windup
        ctrl.integral_att = ctrl.integral_att + e_att * dt;
        for i = 1:3
            if abs(ctrl.integral_att(i)) > ctrl.integral_limit_att(i)
                ctrl.integral_att(i) = sign(ctrl.integral_att(i)) * ctrl.integral_limit_att(i);
            end
        end
        
        % Angular-rate damping. This is much less noisy than finite
        % differencing attitude error.
        if ~isfield(ctrl, 'filtered_omega_error') || isempty(ctrl.filtered_omega_error)
            ctrl.filtered_omega_error = zeros(3, 1);
        end
        ctrl.filtered_omega_error = ctrl.alpha * ctrl.filtered_omega_error + (1 - ctrl.alpha) * e_omega;
        
        % Attitude control law
        tau_cmd = ctrl.kp_att .* e_att + ...
                  ctrl.ki_att .* ctrl.integral_att + ...
                  ctrl.kd_att .* ctrl.filtered_omega_error;
        
        % Apply torque constraints
        if isfield(params, 'torque_max')
            for i = 1:3
                tau_cmd(i) = max(-params.torque_max, min(params.torque_max, tau_cmd(i)));
            end
        end
        
        % Construct control vector
        u = [thrust_cmd; tau_cmd];
        
        % Update states
        ctrl.prev_error_pos = e_pos;
        ctrl.prev_error_att = e_att;
        
        % Validate output
        if any(~isfinite(u))
            ctrl.last_failure = true;
            warning('PID controller output contains non-finite values');
            u = [params.m * params.g; 0; 0; 0];
        end
        
    catch ME
        ctrl.last_failure = true;
        warning('PID controller computation failed: %s', ME.message);
        u = [params.m * params.g; 0; 0; 0];
        ctrl.integral_pos = zeros(3, 1);
        ctrl.integral_att = zeros(3, 1);
    end
end