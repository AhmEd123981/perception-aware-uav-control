function out = onnx_worker(action, varargin)
%ONNX_WORKER Manage a persistent Python ONNX Runtime worker.
%   worker = onnx_worker("start", cfg) starts or reuses a background worker.
%   response = onnx_worker("infer", worker, image_path, mask_path, image_size)
%   onnx_worker("stop", worker) requests shutdown.

persistent worker_state
action = string(action);

switch action
    case "start"
        cfg = varargin{1};
        if isempty(worker_state) || ~is_worker_alive(worker_state)
            worker_state = start_worker(cfg);
        end
        out = worker_state;
    case "infer"
        worker = varargin{1};
        image_path = varargin{2};
        mask_path = varargin{3};
        image_size = varargin{4};
        out = infer_with_worker(worker, image_path, mask_path, image_size);
    case "stop"
        if ~isempty(varargin)
            worker = varargin{1};
        else
            worker = worker_state;
        end
        stop_worker(worker);
        worker_state = [];
        out = [];
    otherwise
        error("onnx_worker:UnknownAction", "Unknown worker action: %s", action);
end
end

function worker = start_worker(cfg)
root_dir = fileparts(fileparts(mfilename("fullpath")));
request_dir = fullfile(char(cfg.output_root), "onnx_worker");
if ~exist(request_dir, "dir")
    mkdir(request_dir);
end
ready_file = fullfile(request_dir, "ready.json");
stop_file = fullfile(request_dir, "stop.flag");
log_file = fullfile(request_dir, "worker.log");
delete_if_exists(ready_file);
delete_if_exists(stop_file);

model_path = fullfile(root_dir, char(cfg.model.onnx_path));
script_path = fullfile(root_dir, "deep_learning", "infer_worker.py");
python_exe = "python";
if isfield(cfg.model, "python_executable")
    python_exe = cfg.model.python_executable;
end

cmd = sprintf('cmd /c start "" /B %s %s --model %s --request-dir %s --ready-file %s --stop-file %s --log-file %s', ...
    quote_path(python_exe), quote_path(script_path), quote_path(model_path), ...
    quote_path(request_dir), quote_path(ready_file), quote_path(stop_file), quote_path(log_file));
[status, output] = system(cmd);
if status ~= 0
    error("onnx_worker:StartFailed", "Failed to start ONNX worker:\n%s", output);
end

deadline = tic;
while ~isfile(ready_file)
    pause(0.05);
    if toc(deadline) > 30
        error("onnx_worker:ReadyTimeout", "ONNX worker did not become ready. Log: %s", log_file);
    end
end

worker = struct();
worker.request_dir = request_dir;
worker.ready_file = ready_file;
worker.stop_file = stop_file;
worker.log_file = log_file;
worker.sequence = 0;
worker.model_path = model_path;
end

function response = infer_with_worker(worker, image_path, mask_path, image_size)
request_id = char(java.util.UUID.randomUUID());
request_path = fullfile(worker.request_dir, sprintf("request_%s.json", request_id));
response_path = fullfile(worker.request_dir, sprintf("response_%s.json", request_id));
payload = struct();
payload.id = request_id;
payload.image = char(image_path);
payload.output_mask = char(mask_path);
payload.image_size = image_size;
payload.response = response_path;
write_json(request_path, payload);

deadline = tic;
while ~isfile(response_path)
    pause(0.01);
    if toc(deadline) > 60
        error("onnx_worker:InferenceTimeout", "ONNX worker timed out for request %s.", request_id);
    end
end

response = read_response_when_ready(response_path);
if isfield(response, "ok") && ~response.ok
    error("onnx_worker:InferenceFailed", "ONNX worker failed: %s", response.message);
end
if ~isfile(mask_path)
    error("onnx_worker:MissingMask", "ONNX worker response was ready but mask was not created: %s", mask_path);
end
delete_if_exists(response_path);
end

function response = read_response_when_ready(response_path)
deadline = tic;
last_error = [];
while toc(deadline) <= 10
    if isfile(response_path)
        try
            text = fileread(response_path);
            if ~isempty(strtrim(text))
                response = jsondecode(text);
                return;
            end
        catch err
            last_error = err;
        end
    end
    pause(0.02);
end
if isempty(last_error)
    error("onnx_worker:ResponseReadFailed", ...
        "ONNX worker response file was not readable: %s", response_path);
else
    error("onnx_worker:ResponseReadFailed", ...
        "ONNX worker response file was not readable: %s\n%s", response_path, last_error.message);
end
end

function stop_worker(worker)
if isempty(worker) || ~isfield(worker, "stop_file")
    return;
end
fid = fopen(worker.stop_file, "w");
if fid >= 0
    fprintf(fid, "stop\n");
    fclose(fid);
end
end

function alive = is_worker_alive(worker)
alive = isstruct(worker) && isfield(worker, "ready_file") && isfile(worker.ready_file) && ...
        isfield(worker, "stop_file") && ~isfile(worker.stop_file);
end

function write_json(path_out, data)
tmp_path = [char(path_out), '.tmp'];
fid = fopen(tmp_path, "w");
if fid < 0
    error("onnx_worker:WriteFailed", "Could not write request file: %s", tmp_path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "%s", jsonencode(data));
clear cleanup;
movefile(tmp_path, path_out, "f");
end

function delete_if_exists(path_in)
if isfile(path_in)
    try
        delete(path_in);
    catch
    end
end
end

function quoted = quote_path(path_in)
path_in = char(path_in);
quoted = ['"', strrep(path_in, '"', '\"'), '"'];
end
