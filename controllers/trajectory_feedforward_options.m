function options = trajectory_feedforward_options(ctrl)
    %TRAJECTORY_FEEDFORWARD_OPTIONS Extract flatness-reference limits from a controller.
    options = struct();
    options.acc_xy_limit = get_ctrl_field(ctrl, 'ff_acc_xy_limit', 4.0);
    options.acc_z_limit  = get_ctrl_field(ctrl, 'ff_acc_z_limit',  3.0);
    options.tilt_limit   = get_ctrl_field(ctrl, 'ff_tilt_limit',   25*pi/180);

    params = [];
    if isfield(ctrl, 'params')
        params = ctrl.params;
    end
    if ~isempty(params) && isfield(params, 'm') && isfield(params, 'g')
        mg = params.m * params.g;
        options.min_thrust = get_ctrl_field(ctrl, 'ff_min_thrust', 0.20 * mg);
        options.max_thrust = get_ctrl_field(ctrl, 'ff_max_thrust', 2.00 * mg);
    end
end

function value = get_ctrl_field(ctrl, name, default_value)
    if isfield(ctrl, name) && ~isempty(ctrl.(name))
        value = ctrl.(name);
    else
        value = default_value;
    end
end
