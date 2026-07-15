function world_xy = pixel_to_world(pixel_uv, K, R_uav, t_uav, ground_z)
%PIXEL_TO_WORLD Project image pixels to a flat ground plane.
%   world_xy = pixel_to_world(pixel_uv, K, R_uav, t_uav, ground_z) projects
%   Nx2 pixel coordinates [u v] into world XY coordinates using a pinhole
%   model and known camera pose. R_uav is interpreted as camera-to-world
%   rotation when used by the perception pipeline.
%
%   Pixel coordinates are one-based MATLAB image coordinates. The function
%   returns NaN for rays parallel to the ground or intersecting behind camera.

arguments
    pixel_uv (:,2) double
    K (3,3) double
    R_uav (3,3) double
    t_uav (3,1) double
    ground_z (1,1) double = 0.0
end

num_pixels = size(pixel_uv, 1);
world_xy = nan(num_pixels, 2);

if num_pixels == 0
    return;
end

uv1 = [pixel_uv(:, 1)'; pixel_uv(:, 2)'; ones(1, num_pixels)];
rays_cam = K \ uv1;
rays_cam = rays_cam ./ vecnorm(rays_cam);
rays_world = R_uav * rays_cam;

origin = t_uav(:);
den = rays_world(3, :);
valid = abs(den) > 1e-9;
lambda = nan(1, num_pixels);
lambda(valid) = (ground_z - origin(3)) ./ den(valid);
valid = valid & lambda > 0;

points = origin + rays_world .* lambda;
world_xy(valid, :) = points(1:2, valid)';
end
