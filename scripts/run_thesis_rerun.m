function run_thesis_rerun(num_mc)
%RUN_THESIS_RERUN One-command re-run of the fixed real-GT perception pipeline.
%   run_thesis_rerun() runs, in order:
%     1. day8_real_field_perception            (real-field GT recall check)
%     2. day6_controller_comparison_demo        (agricultural, 5 Hz)
%     3. day6_controller_comparison_circular    (circular,     5 Hz)
%   then refreshes the LaTeX tables (analysis/thesis_analysis.py) and the Day-7
%   thesis package (scripts/day7_finalize_thesis_package.py). It prints the
%   per-controller recall after each Day-6 run so you can confirm recall is now
%   non-zero and controller-dependent.
%
%   run_thesis_rerun(N) also runs perception_monte_carlo(N) before the refresh.
%   WARNING: at 5 Hz the Monte Carlo captures ~10x more frames than the old
%   0.5 Hz runs, so it is slow (roughly half an hour per run pair). Leave num_mc
%   at 0 for a quick pass; run run_thesis_rerun(5) overnight for the MC table.
%
%   Designed to be launched with a single word from a phone remote-desktop
%   session. Everything writes under results/perception_logs and results/analysis.

if nargin < 1 || isempty(num_mc)
    num_mc = 0;
end

repo_root = fileparts(fileparts(mfilename("fullpath")));
cd(repo_root);
addpath(genpath(repo_root));

banner = @(varargin) fprintf("\n==================== %s ====================\n", sprintf(varargin{:}));
t_all = tic;

banner("STEP 1  day8 real-field recall");
try
    day8_real_field_perception();
catch e
    fprintf(2, "day8 FAILED: %s\n", e.message);
end

banner("STEP 2  Day-6 lawnmower (agricultural) @ 5 Hz");
try
    show_recall(day6_controller_comparison_demo("CameraFps", 5.0), "agricultural");
catch e
    fprintf(2, "Day-6 agricultural FAILED: %s\n", e.message);
end

banner("STEP 3  Day-6 circular @ 5 Hz");
try
    show_recall(day6_controller_comparison_circular("CameraFps", 5.0), "circular");
catch e
    fprintf(2, "Day-6 circular FAILED: %s\n", e.message);
end

if num_mc > 0
    banner("STEP 4  Monte Carlo n=%d (SLOW)", num_mc);
    try
        disp(perception_monte_carlo(num_mc));
    catch e
        fprintf(2, "Monte Carlo FAILED: %s\n", e.message);
    end
end

banner("REFRESH  LaTeX tables + Day-7 package (python)");
run_python("analysis/thesis_analysis.py");
run_python("scripts/day7_finalize_thesis_package.py");

fprintf("\nALL DONE in %.1f min.\n", toc(t_all) / 60);
fprintf("Check: recall should be NON-ZERO and DIFFER across PID/LQR/Hinf/MPC/SMC.\n");
fprintf("Tables refreshed: results/analysis/thesis_tables.tex\n");
fprintf("Package refreshed: results/perception_logs/day7_thesis_package/\n");
end

function show_recall(summary, name)
m = summary.metrics;
fprintf("Recall by controller (%s):\n", name);
for i = 1:height(m)
    fprintf("  %-5s  recall=%.3f  coverage=%.1f%%  false_alarm=%.5f  latency=%.2fs\n", ...
        char(m.controller(i)), m.affected_zone_recall(i), m.coverage_percent(i), ...
        m.false_alarm_rate_per_m2(i), m.mean_latency_s(i));
end
end

function run_python(script_rel)
[status, out] = system(sprintf('python "%s"', script_rel));
if status ~= 0
    fprintf(2, "python step failed: %s\n", script_rel);
    fprintf(2, "run it yourself in a terminal:  python %s\n%s\n", script_rel, out);
else
    fprintf("ok: %s\n", script_rel);
end
end
