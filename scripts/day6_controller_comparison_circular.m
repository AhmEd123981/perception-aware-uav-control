function summary = day6_controller_comparison_circular(varargin)
%DAY6_CONTROLLER_COMPARISON_CIRCULAR Run Day 6 circular perception comparison.
%   The circular case uses the same controller/perception integration as the
%   lawnmower case, but requests the circular trajectory generator and writes
%   thesis artifacts under results/perception_logs/day6_circular.

repo_root = fileparts(fileparts(mfilename("fullpath")));
old_path = path;
cleanup = onCleanup(@() path(old_path));
addpath(genpath(repo_root));

output_root = fullfile("results", "perception_logs", "day6_circular");
summary = main_comparison_with_perception("circular", "OutputRoot", output_root, varargin{:});

source_csv = fullfile(output_root, "circular", "full_integration_metrics.csv");
target_csv = fullfile(output_root, "day6_controller_metrics.csv");
if isfile(source_csv)
    copyfile(source_csv, target_csv);
end

source_plot = fullfile(output_root, "circular", "latency_comparison.png");
target_plot = fullfile(output_root, "day6_latency_comparison.png");
if isfile(source_plot)
    copyfile(source_plot, target_plot);
end
end
