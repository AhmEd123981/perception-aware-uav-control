function varargout = controller_adapter(action, varargin)
%CONTROLLER_ADAPTER Thin adapter around the original thesis controllers.
%   This file deliberately does not implement any control law. It validates
%   and calls the existing thesis API:
%
%       ctrl = controller_<NAME>(params, dt, options)
%       u    = controller_<NAME>_compute(uav, x_ref, ctrl, params, math)
%
%   The adapter exists only to keep perception integration code independent
%   from controller file naming details and to print exactly which original
%   controller files are used in the full integration run.

switch string(action)
    case "list"
        varargout{1} = build_controller_specs();
    case "validate"
        if nargin >= 2 && ~isempty(varargin{1})
            names = string(varargin{1});
        else
            names = ["PID", "LQR", "Hinf", "MPC", "SMC"];
        end
        varargout{1} = validate_controller_specs(names);
    case "init"
        varargout{1} = initialize_controller(varargin{:});
    case "compute"
        [varargout{1:nargout}] = compute_controller_step(varargin{:});
    otherwise
        error("controller_adapter:UnknownAction", "Unknown action: %s", action);
end
end

function specs = build_controller_specs()
names = ["PID", "LQR", "Hinf", "MPC", "SMC"];
specs = repmat(struct( ...
    "name", "", ...
    "init_function", "", ...
    "compute_function", "", ...
    "init_file", "", ...
    "compute_file", ""), numel(names), 1);

for i = 1:numel(names)
    name = char(names(i));
    specs(i).name = name;
    specs(i).init_function = sprintf("controller_%s", name);
    specs(i).compute_function = sprintf("controller_%s_compute", name);
    specs(i).init_file = fullfile("controllers", sprintf("controller_%s.m", name));
    specs(i).compute_file = fullfile("controllers", sprintf("controller_%s_compute.m", name));
end
end

function specs = validate_controller_specs(names)
all_specs = build_controller_specs();
keep = ismember(string({all_specs.name}), names);
specs = all_specs(keep);
missing = strings(0, 1);

for i = 1:numel(specs)
    if ~isfile(specs(i).init_file)
        missing(end + 1, 1) = string(specs(i).init_file); %#ok<AGROW>
    end
    if ~isfile(specs(i).compute_file)
        missing(end + 1, 1) = string(specs(i).compute_file); %#ok<AGROW>
    end
    if exist(specs(i).init_function, "file") ~= 2
        missing(end + 1, 1) = string(specs(i).init_function) + " on MATLAB path"; %#ok<AGROW>
    end
    if exist(specs(i).compute_function, "file") ~= 2
        missing(end + 1, 1) = string(specs(i).compute_function) + " on MATLAB path"; %#ok<AGROW>
    end
end

if ~isempty(missing)
    message = sprintf("  - %s\n", strjoin(missing, string(newline) + "  - "));
    error("controller_adapter:MissingControllerAPI", ...
        "The original thesis controller API is not available in this repo root.\n%s", message);
end

fprintf("Original controller files selected for perception integration:\n");
for i = 1:numel(specs)
    fprintf("  %s init:    %s\n", specs(i).name, specs(i).init_file);
    fprintf("  %s compute: %s\n", specs(i).name, specs(i).compute_file);
end
end

function ctrl = initialize_controller(spec, params, dt, options)
if isstruct(spec)
    init_function = spec.init_function;
else
    init_function = sprintf("controller_%s", char(spec));
end
ctrl = feval(init_function, params, dt, options);
end

function varargout = compute_controller_step(spec, uav, x_ref, ctrl, params, math)
if isstruct(spec)
    compute_function = spec.compute_function;
else
    compute_function = sprintf("controller_%s_compute", char(spec));
end
if ~isa(x_ref, "function_handle")
    x_ref = x_ref(:);
end
if nargout >= 2
    [varargout{1}, varargout{2}] = feval(compute_function, uav, x_ref, ctrl, params, math);
else
    varargout{1} = feval(compute_function, uav, x_ref, ctrl, params, math);
end
end
