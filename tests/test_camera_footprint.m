function test_camera_footprint()
%TEST_CAMERA_FOOTPRINT Verify camera geometry at 4 m spray altitude.
%   The test projects image corners to the ground plane and checks the
%   measured footprint against the configured horizontal and vertical FOV.

cfg = perception_config();
cfg.enabled = true;
cfg.save_frames = false;
cfg.output_root = fullfile("results", "perception_logs");
cfg.seed = 2026;

uav_state = struct();
uav_state.position = [7.5; 5.0; 4.0];
uav_state.R_world_body = eye(3);
uav_state.velocity = [0; 0; 0];

field_model = struct();
field_model.row_period_m = 0.75;
field_model.row_width_m = 0.35;
field_model.affected_patches = struct("center", {[7.5, 5.0]}, ...
                                      "radius", {[0.75, 0.50]}, ...
                                      "class_id", {3});

frame = camera_simulator(uav_state, field_model, cfg, 0.0, 1);

height = cfg.camera.resolution(1);
width = cfg.camera.resolution(2);
corners_uv = [1, 1; width, 1; width, height; 1, height];
corners_xy = pixel_to_world(corners_uv, frame.K, frame.R_world_cam, frame.t_world_cam, cfg.field.ground_z_m);

actual_width = 0.5 * (norm(corners_xy(2, :) - corners_xy(1, :)) + ...
                      norm(corners_xy(3, :) - corners_xy(4, :)));
actual_height = 0.5 * (norm(corners_xy(4, :) - corners_xy(1, :)) + ...
                       norm(corners_xy(3, :) - corners_xy(2, :)));
expected_width = 2 * uav_state.position(3) * tand(cfg.camera.horizontal_fov_deg / 2);
expected_height = 2 * uav_state.position(3) * tand(cfg.camera.vertical_fov_deg / 2);

fprintf("Camera footprint sanity check\n");
fprintf("Expected width: %.3f m | actual width: %.3f m\n", expected_width, actual_width);
fprintf("Expected height: %.3f m | actual height: %.3f m\n", expected_height, actual_height);

width_error = abs(actual_width - expected_width) / expected_width;
height_error = abs(actual_height - expected_height) / expected_height;
center_error = norm(mean(corners_xy, 1) - uav_state.position(1:2)');
mirrored_y = corners_xy(1, 2) < uav_state.position(2);

if width_error > 0.02 || height_error > 0.02 || center_error > 1e-2 || mirrored_y
    warning("test_camera_footprint:Convention", ...
        ["Camera footprint convention may be wrong. Recommended level downward " ...
         "camera replacement is R_body_cam = [1 0 0; 0 -1 0; 0 0 -1]."]);
end

assert(width_error <= 0.02, "Footprint width differs from expected FOV by more than 2%%.");
assert(height_error <= 0.02, "Footprint height differs from expected FOV by more than 2%%.");
assert(center_error <= 1e-2, "Camera footprint is not centered under the UAV.");

out_dir = fullfile(cfg.output_root, "sanity");
if ~exist(out_dir, "dir")
    mkdir(out_dir);
end

fig = figure("Visible", "off", "Color", "w");
hold on;
rectangle("Position", [0, 0, cfg.field.length_m, cfg.field.width_m], ...
          "EdgeColor", [0.1, 0.1, 0.1], "LineWidth", 1.5);
patch(corners_xy(:, 1), corners_xy(:, 2), [0.2, 0.5, 0.9], ...
      "FaceAlpha", 0.25, "EdgeColor", [0.0, 0.2, 0.8], "LineWidth", 2.0);
plot(uav_state.position(1), uav_state.position(2), "ko", "MarkerFaceColor", "k");
text(uav_state.position(1) + 0.1, uav_state.position(2), "UAV", "FontWeight", "bold");
axis equal;
xlim([-0.5, cfg.field.length_m + 0.5]);
ylim([-0.5, cfg.field.width_m + 0.5]);
grid on;
xlabel("x world [m]");
ylabel("y world [m]");
title("Downward Camera Ground Footprint at 4 m");
exportgraphics(fig, fullfile(out_dir, "camera_footprint.png"), "Resolution", 200);
close(fig);

fprintf("Saved footprint plot: %s\n", fullfile(out_dir, "camera_footprint.png"));
end
