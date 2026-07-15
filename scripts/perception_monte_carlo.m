function aggregate = perception_monte_carlo(num_runs, varargin)
%PERCEPTION_MONTE_CARLO Run controller/trajectory perception MC analysis.
%   perception_monte_carlo(10) runs agricultural and circular trajectories
%   through main_comparison_with_perception with deterministic but distinct
%   seeds for wind, initial conditions, and camera/rendering noise. The
%   original controller and dynamics files are used through the main
%   integration entry point.

if nargin < 1 || isempty(num_runs)
    num_runs = 10;
end

parser = inputParser();
% Duration must be long enough for the UAV to fly the full lawnmower survey
% over the 15x10 m field. The earlier 12 s default only covered ~25% of the
% field, which collapsed affected-zone recall to ~0; 65 s matches the Day-6
% deterministic runs and reproduces all five controllers correctly.
parser.addParameter("Duration", 65.0);
% 5 Hz matches the camera spec (cfg.camera.fps) and the deterministic Day-6
% runs. The earlier 0.5 Hz (1 frame / 2 s) under-sampled the field so heavily
% that every controller produced near-identical coverage/recall; at 5 Hz the
% footprint overlap depends on how well each controller tracks the lawnmower,
% so the five controllers separate again.
parser.addParameter("CameraFps", 5.0);
parser.addParameter("ImageSize", 256);  % av_segmentation_2020.onnx has fixed 256x256 input (was 512 for the old synthetic model)
parser.parse(varargin{:});
opts = parser.Results;

output_root = fullfile("results", "perception_logs", "monte_carlo");
if ~isfolder(output_root)
    mkdir(output_root);
end

trajectories = ["agricultural", "circular"];
raw_metrics = table();

for run_id = 1:num_runs
    for traj = trajectories
        seed_base = 202600 + 1000 * run_id + 17 * find(trajectories == traj, 1);
        run_root = fullfile(output_root, sprintf("run_%02d", run_id));
        fprintf("\nMonte Carlo run %d/%d trajectory=%s seed=%d\n", ...
            run_id, num_runs, traj, seed_base);
        summary = main_comparison_with_perception(traj, ...
            "Seed", seed_base, ...
            "WindSeed", seed_base + 101, ...
            "InitialConditionSeed", seed_base + 202, ...
            "Duration", opts.Duration, ...
            "CameraFps", opts.CameraFps, ...
            "ImageSize", opts.ImageSize, ...
            "OutputRoot", run_root);

        T = summary.metrics;
        T.run_id = repmat(run_id, height(T), 1);
        T.trajectory = repmat(traj, height(T), 1);
        T.seed = repmat(seed_base, height(T), 1);
        raw_metrics = [raw_metrics; T]; %#ok<AGROW>
    end
end

raw_csv = fullfile(output_root, "perception_monte_carlo_raw.csv");
writetable(raw_metrics, raw_csv);
aggregate = aggregate_metrics(raw_metrics);
aggregate_csv = fullfile(output_root, "perception_monte_carlo_aggregate.csv");
writetable(aggregate, aggregate_csv);
summary_csv = fullfile(output_root, "summary.csv");
writetable(to_required_summary_schema(aggregate), summary_csv);
plot_monte_carlo_summary(aggregate, fullfile("results", "figures", "perception_monte_carlo_summary.png"));
save(fullfile(output_root, "perception_monte_carlo_summary.mat"), "raw_metrics", "aggregate");

fprintf("\nMonte Carlo perception analysis complete.\n");
fprintf("Raw metrics: %s\n", raw_csv);
fprintf("Aggregate metrics: %s\n", aggregate_csv);
fprintf("Required summary: %s\n", summary_csv);
end

function aggregate = aggregate_metrics(raw_metrics)
metric_names = ["coverage_percent", "affected_zone_recall", ...
    "mean_latency_s", "false_alarm_rate_per_m2"];
controllers = unique(raw_metrics.controller, "stable");
trajectories = unique(raw_metrics.trajectory, "stable");

rows = {};
for i = 1:numel(trajectories)
    for j = 1:numel(controllers)
        mask = raw_metrics.trajectory == trajectories(i) & raw_metrics.controller == controllers(j);
        row = struct();
        row.trajectory = trajectories(i);
        row.controller = controllers(j);
        row.n = nnz(mask);
        for metric = metric_names
            base_name = char(metric);
            values = raw_metrics.(base_name)(mask);
            values = values(isfinite(values));
            if isempty(values)
                mu = NaN;
                sigma = NaN;
                ci95 = NaN;
            else
                mu = mean(values);
                sigma = std(values);
                ci95 = 1.96 * sigma / sqrt(numel(values));
            end
            row.([base_name, '_mean']) = mu;
            row.([base_name, '_std']) = sigma;
            row.([base_name, '_ci95']) = ci95;
        end
        rows{end + 1, 1} = row; %#ok<AGROW>
    end
end
aggregate = struct2table(vertcat(rows{:}));
end

function summary = to_required_summary_schema(aggregate)
summary = table();
summary.controller = aggregate.controller;
summary.trajectory = aggregate.trajectory;
summary.mean_coverage = aggregate.coverage_percent_mean;
summary.std_coverage = aggregate.coverage_percent_std;
summary.mean_recall = aggregate.affected_zone_recall_mean;
summary.std_recall = aggregate.affected_zone_recall_std;
summary.mean_latency_s = aggregate.mean_latency_s_mean;
summary.std_latency_s = aggregate.mean_latency_s_std;
summary.mean_false_alarm = aggregate.false_alarm_rate_per_m2_mean;
summary.std_false_alarm = aggregate.false_alarm_rate_per_m2_std;
summary = movevars(summary, "trajectory", "After", "controller");
end

function plot_monte_carlo_summary(aggregate, output_path)
if ~isfolder(fileparts(output_path))
    mkdir(fileparts(output_path));
end
metrics = ["coverage_percent", "affected_zone_recall", "mean_latency_s", "false_alarm_rate_per_m2"];
titles = ["Coverage [%]", "Affected-Zone Recall", "Mean Latency [s]", "False Alarm Area Rate"];
controllers = unique(aggregate.controller, "stable");
trajectories = unique(aggregate.trajectory, "stable");

fig = figure("Visible", "off", "Color", "w", "Position", [100, 100, 1200, 720]);
tiledlayout(2, 2, "Padding", "compact", "TileSpacing", "compact");
for m = 1:numel(metrics)
    nexttile;
    values = nan(numel(controllers), numel(trajectories));
    ci = nan(size(values));
    for i = 1:numel(controllers)
        for j = 1:numel(trajectories)
            row = aggregate.controller == controllers(i) & aggregate.trajectory == trajectories(j);
            if any(row)
                metric_name = char(metrics(m));
                mean_values = aggregate.([metric_name, '_mean']);
                ci_values = aggregate.([metric_name, '_ci95']);
                values(i, j) = mean_values(row);
                ci(i, j) = ci_values(row);
            end
        end
    end
    b = bar(categorical(controllers), values);
    hold on;
    for j = 1:numel(trajectories)
        x = b(j).XEndPoints;
        errorbar(x, values(:, j), ci(:, j), "k.", "LineWidth", 1.0);
    end
    title(titles(m));
    grid on;
    legend(trajectories, "Location", "best");
end
exportgraphics(fig, output_path, "Resolution", 200);
close(fig);
end
