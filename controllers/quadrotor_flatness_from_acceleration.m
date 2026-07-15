function ff = quadrotor_flatness_from_acceleration(x_ref, a_ref, params, math, options)
    %QUADROTOR_FLATNESS_FROM_ACCELERATION
    % Convert trajectory acceleration into nominal quadrotor thrust/attitude.
    % This is a feedforward reference generator only; feedback remains inside
    % PID/LQR/Hinf/MPC/SMC.

    if nargin < 5 || isempty(options)
        options = struct();
    end

    x_ref = x_ref(:);
    a_ref = a_ref(:);
    if numel(a_ref) < 3
        a_ref = zeros(3, 1);
    else
        a_ref = a_ref(1:3);
    end
    a_ref(~isfinite(a_ref)) = 0;

    g = params.g;
    mg = params.m * params.g;

    acc_xy_limit = get_option(options, 'acc_xy_limit', 4.0);
    acc_z_limit  = get_option(options, 'acc_z_limit',  3.0);
    tilt_limit   = get_option(options, 'tilt_limit',   25*pi/180);
    min_thrust   = get_option(options, 'min_thrust',   0.20 * mg);
    max_thrust   = get_option(options, 'max_thrust',   2.00 * mg);

    % Keep the nominal reference inside the hover-linear model validity region.
    a_xy = a_ref(1:2);
    a_xy_norm = norm(a_xy);
    if a_xy_norm > acc_xy_limit
        a_ref(1:2) = a_xy * (acc_xy_limit / max(a_xy_norm, eps));
    end
    a_ref(3) = max(-acc_z_limit, min(acc_z_limit, a_ref(3)));

    vertical_acc = max(0.2 * g, g + a_ref(3));
    max_xy_from_tilt = vertical_acc * tan(tilt_limit);
    a_xy_norm = norm(a_ref(1:2));
    if a_xy_norm > max_xy_from_tilt
        a_ref(1:2) = a_ref(1:2) * (max_xy_from_tilt / max(a_xy_norm, eps));
    end

    F_ref = params.m * [a_ref(1); a_ref(2); g + a_ref(3)];
    T_nom = norm(F_ref);
    if T_nom < 1e-9
        F_ref = [0; 0; mg];
        T_nom = mg;
    end

    b3_des = F_ref / T_nom;
    thrust = max(min_thrust, min(max_thrust, T_nom));

    yaw_ref = 0;
    if numel(x_ref) >= 10
        q_ref = x_ref(7:10);
        if all(isfinite(q_ref)) && norm(q_ref) > 1e-9
            try
                R_yaw = math.quat_to_dcm(q_ref / norm(q_ref));
                eul = math.dcm_to_euler(R_yaw);
                yaw_ref = eul(3);
            catch
                yaw_ref = 0;
            end
        end
    end

    x_c = [cos(yaw_ref); sin(yaw_ref); 0];
    b2_des = cross(b3_des, x_c);
    if norm(b2_des) < 1e-8
        x_c = [cos(yaw_ref + pi/2); sin(yaw_ref + pi/2); 0];
        b2_des = cross(b3_des, x_c);
    end
    b2_des = b2_des / (norm(b2_des) + eps);
    b1_des = cross(b2_des, b3_des);
    b1_des = b1_des / (norm(b1_des) + eps);

    R_des = [b1_des, b2_des, b3_des];
    [U, ~, V] = svd(R_des);
    R_des = U * V';
    if det(R_des) < 0
        U(:, 3) = -U(:, 3);
        R_des = U * V';
    end

    try
        euler_ref = math.dcm_to_euler(R_des);
    catch
        euler_ref = local_dcm_to_euler(R_des);
    end

    try
        q_ref = math.dcm_to_quat(R_des);
    catch
        q_ref = local_dcm_to_quat(R_des);
    end
    q_ref = q_ref(:) / (norm(q_ref) + eps);

    ff = struct();
    ff.acc = a_ref;
    ff.force = F_ref;
    ff.thrust = thrust;
    ff.R = R_des;
    ff.euler = euler_ref(:);
    ff.quat = q_ref;
    ff.yaw = yaw_ref;
    ff.valid = true;
end

function value = get_option(options, name, default_value)
    if isfield(options, name) && ~isempty(options.(name))
        value = options.(name);
    else
        value = default_value;
    end
end

function q = local_dcm_to_quat(R)
    tr = trace(R);
    if tr > 0
        S = sqrt(tr + 1.0) * 2;
        qw = 0.25 * S;
        qx = (R(3,2) - R(2,3)) / S;
        qy = (R(1,3) - R(3,1)) / S;
        qz = (R(2,1) - R(1,2)) / S;
    elseif (R(1,1) > R(2,2)) && (R(1,1) > R(3,3))
        S = sqrt(1.0 + R(1,1) - R(2,2) - R(3,3)) * 2;
        qw = (R(3,2) - R(2,3)) / S;
        qx = 0.25 * S;
        qy = (R(1,2) + R(2,1)) / S;
        qz = (R(1,3) + R(3,1)) / S;
    elseif R(2,2) > R(3,3)
        S = sqrt(1.0 + R(2,2) - R(1,1) - R(3,3)) * 2;
        qw = (R(1,3) - R(3,1)) / S;
        qx = (R(1,2) + R(2,1)) / S;
        qy = 0.25 * S;
        qz = (R(2,3) + R(3,2)) / S;
    else
        S = sqrt(1.0 + R(3,3) - R(1,1) - R(2,2)) * 2;
        qw = (R(2,1) - R(1,2)) / S;
        qx = (R(1,3) + R(3,1)) / S;
        qy = (R(2,3) + R(3,2)) / S;
        qz = 0.25 * S;
    end
    q = [qw; qx; qy; qz];
end

function eul = local_dcm_to_euler(R)
    phi = atan2(R(3,2), R(3,3));
    theta = -asin(max(-1, min(1, R(3,1))));
    psi = atan2(R(2,1), R(1,1));
    eul = [phi; theta; psi];
end
