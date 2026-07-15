function plot_runtime_comparison(runtime_csv, output_path)
%PLOT_RUNTIME_COMPARISON Save perception runtime bars for all controllers.
%   plot_runtime_comparison(runtime_csv) reads the CSV produced by the day7
%   thesis package and writes results/figures/runtime_perception_comparison.png.

if nargin < 1 || isempty(runtime_csv)
    runtime_csv = fullfile("results", "perception_logs", "day7_thesis_package", "runtime_profile.csv");
end
if nargin < 2 || isempty(output_path)
    output_path = fullfile("results", "figures", "runtime_perception_comparison.png");
end

if ~exist(runtime_csv, "file")
    error("plot_runtime_comparison:MissingCSV", "Runtime CSV not found: %s", runtime_csv);
end

T = readtable(runtime_csv);
controller = string(T.controller);
values = T.mean_inference_time_ms;

out_dir = fileparts(output_path);
if ~exist(out_dir, "dir")
    mkdir(out_dir);
end

fig = figure("Visible", "off", "Color", "w");
b = bar(categorical(controller), values);
b.FaceColor = [0.20 0.45 0.75];
grid on;
ylabel("Mean perception runtime per frame [ms]");
title("Perception Runtime by Controller");
set(gca, "FontName", "Arial", "FontSize", 10);
for k = 1:numel(values)
    text(k, values(k), sprintf("%.1f", values(k)), ...
        "HorizontalAlignment", "center", "VerticalAlignment", "bottom", "FontSize", 9);
end
exportgraphics(fig, output_path, "Resolution", 200);
close(fig);
end
