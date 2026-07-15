function [rgb, semantic_mask, world_footprint] = render_field(field_model, opts)
%RENDER_FIELD Render synthetic or orthomosaic agricultural imagery.
%   [rgb, mask, footprint] = render_field(field_model, opts) intersects
%   camera rays with the flat field plane and samples either a supplied
%   orthomosaic or a deterministic procedural field.

arguments
    field_model struct
    opts struct
end

height = opts.resolution(1);
width = opts.resolution(2);
K_inv = inv(opts.K);

[uu, vv] = meshgrid(1:width, 1:height);
pixels_h = [uu(:)'; vv(:)'; ones(1, numel(uu))];
rays_cam = K_inv * pixels_h;
rays_cam = rays_cam ./ vecnorm(rays_cam);

rays_world = opts.R_world_cam * rays_cam;
cam_pos = opts.t_world_cam(:);
lambda = (opts.ground_z_m - cam_pos(3)) ./ rays_world(3, :);
world_pts = cam_pos + rays_world .* lambda;

x = reshape(world_pts(1, :), height, width);
y = reshape(world_pts(2, :), height, width);
world_footprint = [min(x(:)), max(x(:)); min(y(:)), max(y(:))];

if isfield(field_model, "texture_rgb") && isfield(field_model, "texture_bounds")
    [rgb, semantic_mask] = sample_orthomosaic(field_model, x, y);
else
    [rgb, semantic_mask] = render_procedural_field(field_model, x, y, opts.seed);
end

if isfield(opts, "apply_motion_blur") && opts.apply_motion_blur
    kernel = fspecial("motion", 5, 0);
    rgb = imfilter(rgb, kernel, "replicate");
end
end

function [rgb, mask] = render_procedural_field(field_model, x, y, seed)
%RENDER_PROCEDURAL_FIELD Build rows, soil, disease, weeds, and stress zones.
rng(seed);
[height, width] = size(x);
rgb = zeros(height, width, 3, "uint8");
mask = ones(height, width, "uint8");

row_period = getfield_or(field_model, "row_period_m", 0.75);
row_width = getfield_or(field_model, "row_width_m", 0.35);
row_phase = mod(y, row_period);
crop = row_phase < row_width;

soil_r = uint8(112 + 25 * rand(height, width));
soil_g = uint8(83 + 25 * rand(height, width));
soil_b = uint8(45 + 10 * rand(height, width));
rgb(:, :, 1) = soil_r;
rgb(:, :, 2) = soil_g;
rgb(:, :, 3) = soil_b;

green = uint8(70 + 65 * rand(height, width));
rgb(:, :, 1) = uint8(double(rgb(:, :, 1)) .* ~crop + 35 .* crop);
rgb(:, :, 2) = uint8(double(rgb(:, :, 2)) .* ~crop + double(green) .* crop);
rgb(:, :, 3) = uint8(double(rgb(:, :, 3)) .* ~crop + 35 .* crop);
mask(crop) = 2;

patches = getfield_or(field_model, "affected_patches", default_patches());
for i = 1:numel(patches)
    p = patches(i);
    d2 = ((x - p.center(1)).^2 / p.radius(1)^2) + ((y - p.center(2)).^2 / p.radius(2)^2);
    region = d2 <= 1;
    mask(region) = uint8(p.class_id);
    rgb = colorize_region(rgb, region, p.class_id);
end
end

function [rgb, mask] = sample_orthomosaic(field_model, x, y)
%SAMPLE_ORTHOMOSAIC Sample a calibrated world-referenced field raster.
%   field_model.texture_bounds is [xmin xmax; ymin ymax]. RGB uses bilinear
%   sampling; masks use nearest-neighbor sampling. Out-of-bounds samples are
%   returned as soil with a neutral brown color.
[height, width] = size(x);
texture_rgb = field_model.texture_rgb;
if isa(texture_rgb, "double") || isa(texture_rgb, "single")
    if max(texture_rgb(:)) <= 1
        texture_rgb = uint8(255 * texture_rgb);
    else
        texture_rgb = uint8(texture_rgb);
    end
end

if isfield(field_model, "texture_mask")
    texture_mask = field_model.texture_mask;
elseif isfield(field_model, "semantic_mask")
    texture_mask = field_model.semantic_mask;
else
    texture_mask = ones(size(texture_rgb, 1), size(texture_rgb, 2), "uint8");
end
texture_mask = uint8(texture_mask);

bounds = field_model.texture_bounds;
x_limits = double(bounds(1, :));
y_limits = double(bounds(2, :));
ref = imref2d(size(texture_rgb, [1 2]), x_limits, y_limits);
[u_intrinsic, v_intrinsic] = worldToIntrinsic(ref, x, y);

inside = u_intrinsic >= 1 & u_intrinsic <= size(texture_rgb, 2) & ...
         v_intrinsic >= 1 & v_intrinsic <= size(texture_rgb, 1);

rgb = zeros(height, width, 3, "uint8");
neutral_brown = uint8([112, 83, 45]);
for c = 1:3
    channel = double(texture_rgb(:, :, c));
    sampled = interp2(channel, u_intrinsic, v_intrinsic, "linear", NaN);
    out = repmat(neutral_brown(c), height, width);
    valid_values = inside & isfinite(sampled);
    out(valid_values) = uint8(max(0, min(255, round(sampled(valid_values)))));
    rgb(:, :, c) = out;
end

sampled_mask = interp2(double(texture_mask), u_intrinsic, v_intrinsic, "nearest", NaN);
mask = ones(height, width, "uint8");
valid_mask = inside & isfinite(sampled_mask);
mask(valid_mask) = uint8(sampled_mask(valid_mask));
end

function value = getfield_or(s, name, default_value)
if isfield(s, name)
    value = s.(name);
else
    value = default_value;
end
end

function patches = default_patches()
patches = struct("center", {[4.0, 3.0], [9.5, 7.0], [12.0, 4.5]}, ...
                 "radius", {[0.9, 0.6], [1.1, 0.8], [0.7, 1.0]}, ...
                 "class_id", {3, 4, 5});
end

function rgb = colorize_region(rgb, region, class_id)
colors = uint8([150 125 35; 40 120 45; 185 175 55]);
idx = max(1, min(3, class_id - 2));
for c = 1:3
    channel = rgb(:, :, c);
    channel(region) = colors(idx, c);
    rgb(:, :, c) = channel;
end
end
