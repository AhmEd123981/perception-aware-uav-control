function results = run_all_tests()
%RUN_ALL_TESTS Execute perception-extension MATLAB unit tests.
%   results = run_all_tests() runs the geometry, mapping, planning, and
%   spray-event tests used by the thesis perception milestone. The function
%   raises an error on the first failed test so MATLAB -batch exits non-zero.

repo_root = fileparts(fileparts(mfilename("fullpath")));
old_path = path;
cleanup = onCleanup(@() path(old_path));
addpath(genpath(repo_root));
cd(repo_root);

test_names = [
    "test_pixel_to_world"
    "test_camera_footprint"
    "test_treatment_map"
    "test_revisit_planner"
    "test_spray_executor"
];

results = table('Size', [numel(test_names), 3], ...
    'VariableTypes', {'string', 'logical', 'string'}, ...
    'VariableNames', {'test', 'passed', 'message'});

fprintf("Running %d MATLAB perception tests...\n", numel(test_names));
for i = 1:numel(test_names)
    name = test_names(i);
    results.test(i) = name;
    try
        fprintf("  [%d/%d] %s\n", i, numel(test_names), name);
        feval(name);
        results.passed(i) = true;
        results.message(i) = "passed";
    catch err
        results.passed(i) = false;
        results.message(i) = string(err.message);
        disp(results);
        rethrow(err);
    end
end

disp(results);
fprintf("All perception tests passed.\n");
end
