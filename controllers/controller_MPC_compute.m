function [u, ctrl] = controller_MPC_compute(uav, x_ref, ctrl, params, math)
    %% MPC Controller Computation — Closed-form batch QP
    % X = Sx*x0 + Su*U
    % min_U  (X - X_ref)' Q_blk (X - X_ref) + U' R_blk U
    % => u_opt = -(H_qp \ f),  f = Su'*Q_blk*(Sx*x0 - X_ref)
    % Constraints applied by clamping (projection onto box).

    ctrl.last_failure = false;
    try
        tic;
        ctrl.total_calls = ctrl.total_calls + 1;

        % Track simulation time
        if ~isfield(ctrl, 't'), ctrl.t = 0; end
        t0    = ctrl.t;
        ctrl.t = ctrl.t + ctrl.dt;

        if isa(x_ref, 'function_handle')
            x_ref_current = x_ref(t0);
        else
            x_ref_current = x_ref;
        end

        % Current state
        x_pos  = uav.x;
        v      = uav.v;
        R_mat  = uav.R;
        omega  = uav.W;
        euler  = math.dcm_to_euler(R_mat);
        x_current = [x_pos; v; euler; omega];

        n = size(ctrl.A, 1);   % 12
        m = size(ctrl.B, 2);   % 4
        N = ctrl.N;

        % ---------------------------------------------------------------
        % Build reference horizon  X_ref ∈ R^{n(N+1)}
        % ---------------------------------------------------------------
        x_ref_horizon = build_reference_horizon(x_ref, N, ctrl.dt, math, t0, params, ctrl);
        X_ref_vec = reshape(x_ref_horizon, n*(N+1), 1);

        % ---------------------------------------------------------------
        % Closed-form unconstrained solution
        % H_qp * U = -f
        % ---------------------------------------------------------------
        free_response = ctrl.Sx * x_current;          % n(N+1) x 1
        f = ctrl.Su' * ctrl.Q_blk * (free_response - X_ref_vec);

        u_opt = -(ctrl.H_qp \ f);                    % mN x 1

        % ---------------------------------------------------------------
        % Box-constraint projection (input constraints)
        % ---------------------------------------------------------------
        U_min = repmat(ctrl.u_min, N, 1);
        U_max = repmat(ctrl.u_max, N, 1);
        u_opt = max(U_min, min(U_max, u_opt));

        % First control in sequence (deviation from hover)
        du = u_opt(1:m);

        % Warm start: shift sequence by one step
        ctrl.u_prev = [u_opt(m+1:end); u_opt(end-m+1:end)];
        ctrl.feasible_solutions = ctrl.feasible_solutions + 1;

        % Add trajectory-aware feedforward to recover absolute command.
        % MPC still computes du as the feedback/prediction correction.
        u_ff = [params.m * params.g; 0; 0; 0];
        if isfield(ctrl, 'use_trajectory_feedforward') && ctrl.use_trajectory_feedforward
            [ff, ctrl] = compute_trajectory_feedforward(x_ref_current, ctrl, params, math);
            blend = 0.7;
            if isfield(ctrl, 'ff_thrust_blend') && ~isempty(ctrl.ff_thrust_blend)
                blend = ctrl.ff_thrust_blend;
            end
            u_ff(1) = params.m * params.g + blend * (ff.thrust - params.m * params.g);
        end
        u    = u_ff + du;

        % Performance monitoring
        solve_time = toc;
        ctrl.solve_times(end+1) = solve_time;
        ctrl.iterations(end+1)  = 1;   % direct solve = 1 step

        if any(~isfinite(u))
            ctrl.last_failure = true;
            warning('MPC output non-finite');
            u = [params.m * params.g; 0; 0; 0];
        end

    catch ME
        ctrl.last_failure = true;
        warning('MPC computation failed: %s', ME.message);
        u = [params.m * params.g; 0; 0; 0];
        ctrl.u_prev = zeros(size(ctrl.B, 2) * ctrl.N, 1);
    end
end

% -----------------------------------------------------------------------
function x_ref_horizon = build_reference_horizon(x_ref, N, dt, math, t0, params, ctrl)
    % Returns 12 x (N+1) reference matrix.
    % Euler reference is derived from desired trajectory acceleration
    % (feedforward attitude), same approach as LQR/Hinf compute.

    x_ref_horizon = zeros(12, N+1);

    for k = 1:N+1
        if isa(x_ref, 'function_handle')
            x_ref_k = x_ref(t0 + (k-1)*dt);
        else
            x_ref_k = x_ref;
        end

        x_ref_pos  = x_ref_k(1:3);
        v_ref      = x_ref_k(4:6);
        q_ref      = x_ref_k(7:10);
        omega_ref  = x_ref_k(11:13);

        % Nominal attitude from trajectory acceleration. This makes the
        % MPC horizon genuinely trajectory-aware instead of tracking each
        % sample as a static hover target.
        R_ref      = math.quat_to_dcm(q_ref);
        euler_ref  = math.dcm_to_euler(R_ref);
        a_ref      = zeros(3, 1);
        if isa(x_ref, 'function_handle') && k <= N
            x_next = x_ref(t0 + k*dt);
            v_next = x_next(4:6);
            a_ref  = (v_next - v_ref) / dt;
        end
        if nargin >= 7 && isfield(ctrl, 'use_trajectory_feedforward') && ctrl.use_trajectory_feedforward
            ff = quadrotor_flatness_from_acceleration(x_ref_k, a_ref, params, math, trajectory_feedforward_options(ctrl));
            euler_ref = ff.euler;
        end

        x_ref_horizon(:, k) = [x_ref_pos; v_ref; euler_ref; omega_ref];
    end
end
