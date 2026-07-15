function summary = day6_controller_comparison_demo(varargin)
%DAY6_CONTROLLER_COMPARISON_DEMO Agricultural full-integration wrapper.
%   This wrapper preserves the Day 6 file name used by earlier scripts while
%   routing execution to main_comparison_with_perception, which calls the
%   original PID, LQR, Hinf, MPC, and SMC controller files directly.

repo_root = fileparts(fileparts(mfilename("fullpath")));
old_path = path;
cleanup = onCleanup(@() path(old_path));
addpath(genpath(repo_root));

output_root = fullfile("results", "perception_logs", "day6_controller_comparison");
summary = main_comparison_with_perception("agricultural", "OutputRoot", output_root, varargin{:});

source_csv = fullfile(output_root, "agricultural", "full_integration_metrics.csv");
target_csv = fullfile(output_root, "day6_controller_metrics.csv");
if isfile(source_csv)
    copyfile(source_csv, target_csv);
end

source_plot = fullfile(output_root, "agricultural", "latency_comparison.png");
target_plot = fullfile(output_root, "day6_latency_comparison.png");
if isfile(source_plot)
    copyfile(source_plot, target_plot);
end
end
