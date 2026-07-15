function [u, ctrl] = controller_SMC_compute(uav, x_ref, ctrl, params, math)
    %% SMC Controller Computation
    % Sliding mode control with boundary-layer chattering reduction.
    % No disturbance observer — SMC's inherent robustness handles it.

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
        q_ref = x_ref(7:10);
        omega_ref = x_ref(11:13);

        R_ref = math.quat_to_dcm(q_ref);

        %% Position Control using SMC

        % Position and velocity errors
        e_pos = x_ref_pos - x_pos;
        e_vel = v_ref - v;

        % Position sliding surface: s_pos = e_vel + lambda_pos * e_pos
        s_pos = e_vel + ctrl.lambda_pos * e_pos;

        % Trajectory feedforward
        if isfield(ctrl, 'use_trajectory_feedforward') && ctrl.use_trajectory_feedforward
            [ff, ctrl] = compute_trajectory_feedforward(x_ref, ctrl, params, math);
            a_ff = ff.acc;
        else
            a_ff = zeros(3, 1);
        end

        % Switching gains
        if ctrl.use_adaptive
            s_pos_norm = abs(s_pos);
            ctrl.k_hat_pos = ctrl.k_hat_pos + ctrl.gamma_pos .* s_pos_norm * ctrl.dt;
            ctrl.k_hat_pos = max(ctrl.k_min, min(ctrl.k_max, ctrl.k_hat_pos));
            k_pos_adaptive = ctrl.k_hat_pos;
        else
            k_pos_adaptive = diag(ctrl.k_pos);
        end

        % SMC switching term with chattering reduction
        switch ctrl.chattering_reduction
            case 'boundary_layer'
                switching_term_pos = zeros(3, 1);
                for i = 1:3
                    if abs(s_pos(i)) > ctrl.phi_pos
                        switching_term_pos(i) = k_pos_adaptive(i) * sign(s_pos(i));
                    else
                        switching_term_pos(i) = k_pos_adaptive(i) * s_pos(i) / ctrl.phi_pos;
                    end
                end
            case 'sigmoid'
                switching_term_pos = k_pos_adaptive .* (2./(1 + exp(-ctrl.sigmoid_slope * s_pos)) - 1);
            case 'tanh'
                switching_term_pos = k_pos_adaptive .* tanh(ctrl.sigmoid_slope * s_pos);
            otherwise
                switching_term_pos = k_pos_adaptive .* sign(s_pos);
        end

        % Total acceleration command (no observer compensation).  Keep the
        % command inside a feasible tilt envelope so SMC switching cannot
        % request attitudes that the plant cannot track.
        a_cmd = a_ff + ctrl.lambda_pos * e_vel + switching_term_pos;
        if isfield(ctrl, 'max_command_acc_z')
            a_cmd(3) = max(-ctrl.max_command_acc_z, min(ctrl.max_command_acc_z, a_cmd(3)));
        end
        if isfield(ctrl, 'max_command_acc_xy')
            a_xy_norm = norm(a_cmd(1:2));
            if a_xy_norm > ctrl.max_command_acc_xy
                a_cmd(1:2) = a_cmd(1:2) * (ctrl.max_command_acc_xy / max(a_xy_norm, eps));
            end
        end
        if isfield(ctrl, 'max_command_tilt')
            vertical_acc = max(0.2 * params.g, params.g + a_cmd(3));
            max_xy_from_tilt = vertical_acc * tan(ctrl.max_command_tilt);
            a_xy_norm = norm(a_cmd(1:2));
            if a_xy_norm > max_xy_from_tilt
                a_cmd(1:2) = a_cmd(1:2) * (max_xy_from_tilt / max(a_xy_norm, eps));
            end
        end

        F_des = params.m * (params.g * [0; 0; 1] + a_cmd);

        % Desired thrust magnitude and direction
        thrust_des = norm(F_des);
        thrust_min = 0.2 * params.m * params.g;
        thrust_max = 1.8 * params.m * params.g;
        thrust_cmd = max(thrust_min, min(thrust_max, thrust_des));

        if thrust_des > 1e-6
            b3_des = F_des / thrust_des;
            b3_des = b3_des / norm(b3_des);
        else
            b3_des = [0; 0; 1];
        end

        %% Attitude Control using SMC

        % Current thrust direction
        b3_current = R(:, 3);

        % Attitude error (rotation vector)
        e_att_vec = cross(b3_current, b3_des);
        if norm(e_att_vec) > 1e-6
            rotation_axis = e_att_vec / norm(e_att_vec);
            rotation_angle = atan2(norm(e_att_vec), max(-1.0, min(1.0, dot(b3_current, b3_des))));
            e_att = rotation_angle * rotation_axis;
        else
            e_att = zeros(3, 1);
        end

        % Angular velocity error
        e_omega = omega_ref - omega;

        % Attitude sliding surface
        s_att = e_omega + ctrl.lambda_att * e_att;

        % Attitude switching gains
        if ctrl.use_adaptive
            s_att_norm = abs(s_att);
            ctrl.k_hat_att = ctrl.k_hat_att + ctrl.gamma_att .* s_att_norm * ctrl.dt;
            ctrl.k_hat_att = max(ctrl.k_min, min(ctrl.k_max, ctrl.k_hat_att));
            k_att_adaptive = ctrl.k_hat_att;
        else
            k_att_adaptive = diag(ctrl.k_att);
        end

        % Attitude switching with chattering reduction
        switch ctrl.chattering_reduction
            case 'boundary_layer'
                switching_term_att = zeros(3, 1);
                for i = 1:3
                    if abs(s_att(i)) > ctrl.phi_att
                        switching_term_att(i) = k_att_adaptive(i) * sign(s_att(i));
                    else
                        switching_term_att(i) = k_att_adaptive(i) * s_att(i) / ctrl.phi_att;
                    end
                end
            case 'sigmoid'
                switching_term_att = k_att_adaptive .* (2./(1 + exp(-ctrl.sigmoid_slope * s_att)) - 1);
            case 'tanh'
                switching_term_att = k_att_adaptive .* tanh(ctrl.sigmoid_slope * s_att);
            otherwise
                switching_term_att = k_att_adaptive .* sign(s_att);
        end

        % Attitude control law with gyroscopic compensation
        J = params.J;
        tau_cmd = J * (ctrl.lambda_att * e_omega + switching_term_att) + ...
                  cross(omega, J * omega);

        % Apply torque constraints
        if isfield(params, 'torque_max')
            for i = 1:3
                tau_cmd(i) = max(-params.torque_max, min(params.torque_max, tau_cmd(i)));
            end
        end

        % Construct control vector
        u = [thrust_cmd; tau_cmd];

        % Performance monitoring
        ctrl.max_sliding_variable = max(ctrl.max_sliding_variable, norm([s_pos; s_att]));

        % Validate output
        if any(~isfinite(u))
            ctrl.last_failure = true;
            warning('SMC controller output contains non-finite values');
            u = [params.m * params.g; 0; 0; 0];
        end

    catch ME
        ctrl.last_failure = true;
        warning('SMC controller computation failed: %s', ME.message);
        u = [params.m * params.g; 0; 0; 0];
    end
end
