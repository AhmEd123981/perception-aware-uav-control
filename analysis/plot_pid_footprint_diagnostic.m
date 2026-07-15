function summary = plot_pid_footprint_diagnostic(run_mat_path, output_path)
%PLOT_PID_FOOTPRINT_DIAGNOSTIC Plot PID camera footprints over affected zones.
%   summary = plot_pid_footprint_diagnostic(run_mat_path, output_path)
%   loads a PID controller_run.mat file, plots every captured camera
%   footprint over the 15 m x 10 m field, overlays ground-truth affected
%   patch ellipses and centroids, and reports per-patch footprint overlap.

if nargin < 1 || isempty(run_mat_path)
    run_mat_path = fullfile("results", "perception_logs", ...
        "day6_controller_comparison", "agricultural", "PID", "controller_run.mat");
end
if nargin < 2 || isempty(output_path)
    output_path = fullfile("results", "perception_logs", ...
        "day6_controller_comparison", "agricultural", "PID", ...
        "pid_footprint_vs_affected_zones.png");
end

data = load(run_mat_path);
frame_logs = data.frame_logs;
patches = default_affected_patches();
if isfield(data, "field_model") && isfield(data.field_model, "affected_patches")
    patches = data.field_model.affected_patches;
end

summary = struct();
summary.run_mat_path = string(run_mat_path);
summary.output_path = string(output_path);
summary.num_frames = numel(frame_logs);
summary.patch_center_covered = false(numel(patches), 1);
summary.patch_ellipse_overlapped = false(numel(patches), 1);
summary.min_footprint_center_distance_m = nan(numel(patches), 1);

fig = figure("Visible", "off", "Color", "w");
hold on;
axis equal;
xlim([0, 15]);
ylim([0, 10]);
rectangle("Position", [0, 0, 15, 10], "EdgeColor", [0.1, 0.1, 0.1], "LineWidth", 1.5);

for j = 1:numel(frame_logs)
    footprint = frame_logs(j).footprint_xy;
    if size(footprint, 1) < 3 || any(~isfinite(footprint), "all")
        continue;
    end
    patch("XData", footprint(:, 1), "YData", footprint(:, 2), ...
        "FaceColor", [0.2, 0.45, 0.9], "FaceAlpha", 0.06, ...
        "EdgeColor", [0.2, 0.45, 0.9], "EdgeAlpha", 0.18);
end

theta = linspace(0, 2*pi, 96);
for i = 1:numel(patches)
    p = patches(i);
    ex = p.center(1) + p.radius(1) * cos(theta);
    ey = p.center(2) + p.radius(2) * sin(theta);
    plot(ex, ey, "r-", "LineWidth", 1.8);
    plot(p.center(1), p.center(2), "kx", "LineWidth", 1.8, "MarkerSize", 8);
    text(p.center(1) + 0.08, p.center(2) + 0.08, sprintf("P%d", i), ...
        "FontSize", 9, "Color", [0.1, 0.1, 0.1]);

    min_dist = inf;
    for j = 1:numel(frame_logs)
        footprint = frame_logs(j).footprint_xy;
        if size(footprint, 1) < 3 || any(~isfinite(footprint), "all")
            continue;
        end
        summary.patch_center_covered(i) = summary.patch_center_covered(i) || ...
            inpolygon(p.center(1), p.center(2), footprint(:, 1), footprint(:, 2));
        [inside, on_edge] = inpolygon(ex, ey, footprint(:, 1), footprint(:, 2));
        summary.patch_ellipse_overlapped(i) = summary.patch_ellipse_overlapped(i) || any(inside | on_edge);
        center = mean(footprint, 1, "omitnan");
        min_dist = min(min_dist, norm(center - p.center));
    end
    summary.min_footprint_center_distance_m(i) = min_dist;
end

title("PID Camera Footprint vs. Affected Zones");
xlabel("x [m]");
ylabel("y [m]");
grid on;
box on;

out_dir = fileparts(output_path);
if ~isfolder(out_dir)
    mkdir(out_dir);
end
exportgraphics(fig, output_path, "Resolution", 180);
close(fig);

fprintf("PID footprint diagnostic: %s\n", output_path);
for i = 1:numel(patches)
    fprintf("  patch %d: center covered=%d, ellipse overlap=%d, min footprint-center distance=%.2f m\n", ...
        i, summary.patch_center_covered(i), summary.patch_ellipse_overlapped(i), ...
        summary.min_footprint_center_distance_m(i));
end
end

function patches = default_affected_patches()
patches = struct( ...
    "center", {[5.8, 1.05], [10.5, 1.05], [4.5, 3.05], [11.5, 5.1], [7.5, 8.0]}, ...
    "radius", {[0.90, 0.45], [0.95, 0.50], [0.90, 0.65], [1.00, 0.80], [1.10, 0.55]}, ...
    "class_id", {3, 4, 5, 3, 4});
end
