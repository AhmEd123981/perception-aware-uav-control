function plot_latency_comparison(metrics, output_path)
%PLOT_LATENCY_COMPARISON Save grouped latency bars for all controllers.
%   plot_latency_comparison(metrics) writes results/figures/latency_comparison.png.
%   metrics may be the table returned by perception_metrics or a compatible
%   struct array with controller and latency fields.

if nargin < 2 || isempty(output_path)
    output_path = fullfile("results", "figures", "latency_comparison.png");
end

if istable(metrics)
    controller = string(metrics.controller);
    values = [metrics.mean_latency_s, metrics.median_latency_s, metrics.max_latency_s];
else
    controller = string({metrics.controller})';
    values = [[metrics.mean_latency_s]', [metrics.median_latency_s]', [metrics.max_latency_s]'];
end

if isempty(controller)
    error("plot_latency_comparison:EmptyMetrics", "No metrics were provided.");
end

out_dir = fileparts(output_path);
if ~exist(out_dir, "dir")
    mkdir(out_dir);
end

fig = figure("Visible", "off", "Color", "w");
bar(categorical(controller), values, "grouped");
grid on;
ylabel("Detection-to-treatment latency [s]");
legend(["Mean", "Median", "Max"], "Location", "northwest");
title("Perception-Aware Revisit Latency by Controller");
set(gca, "FontName", "Arial", "FontSize", 10);
exportgraphics(fig, output_path, "Resolution", 200);
close(fig);
end
