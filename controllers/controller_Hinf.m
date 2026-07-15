function ctrl = controller_Hinf(params, dt, options)
%% H-INFINITY CONTROLLER
% Full-state H-inf design via the bounded-real ARE (continuous time).
% Disturbance observer bandwidth reduced to 3 rad/s for sensor noise.

    if nargin < 2 || isempty(dt), dt = 0.01; end
    if nargin < 3 || isempty(options), options = struct(); end
    verbose = controller_options_verbose(options);

    ctrl        = struct();
    ctrl.type   = 'Hinf';
    ctrl.params = params;
    ctrl.dt     = dt;
    ctrl.verbose = verbose;
    ctrl.use_trajectory_feedforward = true;
    ctrl.ff_acc_xy_limit = 4.0;
    ctrl.ff_acc_z_limit = 3.0;
    ctrl.ff_tilt_limit = 25*pi/180;
    ctrl.ff_filter_alpha = exp(-12 * dt);
    ctrl.ff_min_thrust = 0.20 * params.m * params.g;
    ctrl.ff_max_thrust = 2.00 * params.m * params.g;

    try
        %% 1. Continuous-time linearised model
        [A_c, B_c, A_d, B_d] = get_hinf_system(params, dt);
        ctrl.A_d = A_d;
        ctrl.B_d = B_d;

        n = size(A_c, 1);
        m = size(B_c, 2);

        %% 2. Disturbance model Bw
        Bw = zeros(n, 3);
        Bw(4, 1) = 1 / params.m;
        Bw(5, 2) = 1 / params.m;
        Bw(6, 3) = 1 / params.m;

        %% 3. H-inf weighting matrices
        Q_z = diag([4, 4, 10,   10, 10, 15,   6, 6, 3,   4, 4, 2.5]);
        R_u = diag([2.0, 60.0, 60.0, 80.0]);

        %% 4. Solve H-inf ARE (gamma iteration)
        gamma = 5.0;
        gamma_found = false;

        try
            X0 = care(A_c, B_c, Q_z, R_u);
        catch
            X0 = eye(n);
        end

        for gamma_try = [5.0, 6.0, 8.0, 10.0, 15.0, 20.0]
            try
                X = X0;
                converged = false;

                for iter = 1:30
                    K_cur = R_u \ B_c' * X;
                    A_cl = A_c - B_c * K_cur;
                    Q_iter = Q_z + K_cur' * R_u * K_cur + ...
                             (1/gamma_try^2) * X * (Bw * Bw') * X;
                    X_new = lyap(A_cl', Q_iter);
                    X_new = (X_new + X_new') / 2;

                    if ~all(eig(X_new) > 0)
                        break;
                    end

                    if norm(X_new - X, 'fro') / (1 + norm(X, 'fro')) < 1e-6
                        converged = true;
                        X = X_new;
                        break;
                    end
                    X = X_new;
                end

                if ~converged, continue; end

                K_try = R_u \ B_c' * X;
                cl_eigs = eig(A_c - B_c * K_try);
                if ~all(real(cl_eigs) < 0), continue; end
                cl_eigs_d = eig(A_d - B_d * K_try);
                if max(abs(cl_eigs_d)) >= 1.0, continue; end

                ctrl.K     = K_try;
                gamma       = gamma_try;
                gamma_found = true;
                break;
            catch
                continue;
            end
        end

        ctrl.design_method = 'Hinf-ARE';
        if ~gamma_found
            if verbose
                warning('Hinf ARE did not converge; using LQR fallback.');
            end
            ctrl.design_method = 'LQR-fallback';
            gamma = Inf;
            try
                ctrl.K = lqr(A_c, B_c, Q_z, R_u);
            catch
                ctrl.K = dlqr(A_d, B_d, Q_z, R_u);
            end
            cl_eigs_d = eig(A_d - B_d * ctrl.K);
            if max(abs(cl_eigs_d)) >= 1.0
                ctrl.K = dlqr(A_d, B_d, Q_z, R_u);
            end
        end

        %% 5. Disturbance observer (bandwidth = 3 rad/s)
        ctrl.use_observer  = true;
        ctrl.d_hat         = zeros(3, 1);
        ctrl.v_prev        = zeros(3, 1);
        ctrl.u_prev        = [params.m*params.g; 0; 0; 0];
        ctrl.R_prev        = eye(3);
        ctrl.L_obs         = 3.0;

        %% 6. Integral action (Z-axis)
        ctrl.use_integral   = true;
        ctrl.Ki_pos         = [0.5; 0.5; 1.0];
        ctrl.integral_pos   = zeros(3, 1);
        ctrl.integral_limit = 2.0;

        ctrl.gamma           = gamma;
        ctrl.condition_number = cond(ctrl.K);

        cl_eigs_cont = eig(A_c - B_c * ctrl.K);
        if verbose
            fprintf('H-infinity Controller initialized:\n');
            fprintf('   - Design method          : %s\n', ctrl.design_method);
            if isfinite(gamma)
                fprintf('   - Achieved gamma         : %.2f\n', gamma);
            else
                fprintf('   - Achieved gamma         : fallback (not certified)\n');
            end
            fprintf('   - Gain condition number  : %.2f\n', ctrl.condition_number);
            fprintf('   - Cont. max Re(eig)      : %.4f\n', max(real(cl_eigs_cont)));
            fprintf('   - Disc. max |eig|        : %.4f\n', max(abs(eig(A_d - B_d*ctrl.K))));
        end
    catch ME
        error('H-infinity controller initialization failed: %s', ME.message);
    end
end

function [A_c, B_c, A_d, B_d] = get_hinf_system(params, dt)
    A_c = zeros(12, 12);
    A_c(1:3, 4:6)   = eye(3);
    A_c(4,  8)      =  params.g;
    A_c(5,  7)      = -params.g;
    A_c(7:9, 10:12) = eye(3);

    B_c = zeros(12, 4);
    B_c(6,  1) = 1 / params.m;
    B_c(10, 2) = 1 / params.Jxx;
    B_c(11, 3) = 1 / params.Jyy;
    B_c(12, 4) = 1 / params.Jzz;

    sys_d = c2d(ss(A_c, B_c, [], []), dt);
    A_d   = sys_d.A;
    B_d   = sys_d.B;
end
