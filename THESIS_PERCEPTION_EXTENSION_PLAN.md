# Perception-Aware Agricultural Quadrotor Extension Plan

## 1. Architecture Diagram and Python/MATLAB Bridge Decision

```text
                         +----------------------------------------------+
                         | Existing MATLAB comparison loop              |
                         | trajectories -> controller -> dynamics       |
                         +---------------------+------------------------+
                                               |
                                               v
+------------------+     +---------------------+------------------------+
| Trajectory       | --> | Controller: PID/LQR/Hinf/MPC/SMC             |
| reference        |     | same reference and same disturbance schedule |
+------------------+     +---------------------+------------------------+
                                               |
                                               v
                         +---------------------+------------------------+
                         | Quadrotor dynamics and sensors               |
                         | state: p, v, attitude, rates, timestamp      |
                         +---------------------+------------------------+
                                               |
                                pose every dt  |  frame trigger 5 Hz
                                               v
                         +---------------------+------------------------+
                         | MATLAB camera simulator                      |
                         | pinhole projection over synthetic field      |
                         +---------------------+------------------------+
                                               |
                                               v
                         +---------------------+------------------------+
                         | RGB frame + pose metadata                    |
                         | PNG/.mat, K, R_wc, t_wc, timestamp          |
                         +---------------------+------------------------+
                                               |
                                               v
                         +---------------------+------------------------+
                         | Python ONNX Runtime inference                |
                         | U-Net semantic segmentation mask/probs       |
                         +---------------------+------------------------+
                                               |
                                               v
                         +---------------------+------------------------+
                         | MATLAB image-to-world projection             |
                         | affected pixels -> ground XY                 |
                         +---------------------+------------------------+
                                               |
                                               v
                         +---------------------+------------------------+
                         | Treatment map accumulator                    |
                         | 0.25 m occupancy/log-odds grid              |
                         +---------------------+------------------------+
                                               |
                              after coverage   v
                         +---------------------+------------------------+
                         | Revisit planner                              |
                         | affected centroids -> TSP-style route        |
                         +---------------------+------------------------+
                                               |
                                               v
                         +---------------------+------------------------+
                         | Phase 2 trajectory reference                 |
                         | revisit + spray event logging                |
                         +----------------------------------------------+
```

### Bridge Options

| Option | Strengths | Weaknesses | Fit |
|---|---|---|---|
| MATLAB Deep Learning Toolbox ONNX import | Single-language runtime, less process overhead | ONNX operator compatibility can break; toolbox version dependent | Good if the exported network imports cleanly |
| Python via `system()` and ONNX Runtime | Simple, reproducible, CPU-friendly, minimal MATLAB dependencies | File I/O overhead; not ideal at high frame rates | Best first implementation |
| MATLAB Python engine | Direct function calls, less manual file handling | Environment fragility; harder for thesis reproducibility on another laptop | Useful after first prototype |
| REST microservice | Decoupled, scalable, easy to profile | Extra server lifecycle and networking code | Overkill for one laptop simulation |

### Decision

Use Python ONNX Runtime called from MATLAB via `system()` for the first production-quality thesis implementation. The perception loop runs at 5 Hz or 10 Hz while control remains at 100 Hz (`dt = 0.01 s`), so inference does not need to run every control step. This keeps MATLAB responsible for dynamics, controllers, mapping, metrics, and figures, while Python owns training, model export, and CPU inference. If latency becomes a bottleneck, replace the command-line bridge with MATLAB Python engine using the same `infer.py` logic.

## 2. Full Folder Tree

```text
project_root/
  main_comparison_enhanced.m                  Existing entry point; add opt-in perception hooks only
  controllers/                                Existing 5 controller implementations; unchanged
  dynamics/                                   Existing quadrotor dynamics; unchanged
  trajectories/                               Existing agricultural and circular references; unchanged
  utils/                                      Existing math/logging helpers; unchanged
  analysis/                                   Existing metrics/Monte Carlo/robustness; unchanged
  results/                                    Existing outputs
    figures/                                  Existing and new figures
      detection_overlay_<traj>_<ctrl>.png     New frame-level segmentation overlay
      treatment_map_<traj>_<ctrl>.png         New accumulated affected-zone map
      latency_comparison.png                  New controller latency comparison
    perception_logs/                          New frame, mask, pose, and spray event logs

  configs/
    perception_config.m                       MATLAB config struct for camera/model/map/replanner
    classes.json                              Shared class IDs for MATLAB/Python boundary

  perception/
    camera_simulator.m                        Captures RGB frame and pose metadata from UAV state
    render_field.m                            Procedural field renderer or orthomosaic sampler

  deep_learning/
    generate_synthetic_dataset.py             Creates synthetic RGB/mask dataset and train/val/test split
    train.py                                  PyTorch semantic segmentation training skeleton
    infer.py                                  ONNX Runtime frame inference CLI
    export_onnx.py                            PyTorch checkpoint to ONNX export
    requirements.txt                          Python dependency list
    models/                                   Optional model architecture modules
    weights/                                  `.pt` and `.onnx` model artifacts

  mapping/
    pixel_to_world.m                          Pixel ray intersection with flat ground plane
    treatment_map.m                           Log-odds treatment grid and zone extraction

  replanner/
    revisit_planner.m                         Affected-zone centroid route and spray event plan

  datasets/
    README.md                                 Dataset conventions and class-id note
    synthetic/
      images/                                 Generated synthetic RGB frames
      masks/                                  Generated class-id segmentation masks
      metadata/                               Pose, seed, renderer parameters
    real/
      images/                                 Real UAV frames or orthomosaic crops
      annotations/                            Manual masks or polygon annotations
    splits/
      train.txt                               Training IDs
      val.txt                                 Validation IDs
      test.txt                                Test IDs

  tests/
    test_pixel_to_world.m                     MATLAB unit test for projection math

  docs/
    projection_math.md                        Pixel-to-ground derivation
    calibration_notes.md                      Camera intrinsics/extrinsics notes
```

## 3. Camera Simulator Specification

Use a downward-facing pinhole RGB camera mounted rigidly to the quadrotor body. Recommended baseline parameters are 1280 x 720 resolution, 60 degree horizontal FOV, approximately 37.6 degree vertical FOV, 6.4 mm x 3.6 mm sensor size, and 5.54 mm focal length. These values are realistic for a small agricultural UAV camera and give a useful ground footprint at 4 m spray altitude without making segmentation too easy.

The intrinsic matrix is:

```text
fx = (W / 2) / tan(FOVx / 2)
fy = (H / 2) / tan(FOVy / 2)
cx = (W - 1) / 2
cy = (H - 1) / 2
K  = [fx 0 cx; 0 fy cy; 0 0 1]
```

Synthetic field rendering should start with procedural crop rows over a 15 m x 10 m field. Healthy crop rows are green bands, soil is brown background, and affected patches are ellipses with class-specific texture/color. For a stronger thesis result, a later version can sample a pre-rendered orthomosaic by intersecting camera rays with the ground plane and using image/world raster referencing.

Frame rate recommendation:

- 5 Hz: best default for a normal laptop; enough for 1 m/s agricultural survey flight and keeps inference off the 100 Hz control loop.
- 10 Hz: better latency and overlap; use when CPU inference is below 50 ms per frame.
- 100 Hz: not recommended; it couples perception cost to the dynamics time step and gives redundant frames.

Each captured frame should save:

- RGB frame as PNG.
- Optional semantic ground truth mask during simulation.
- Pose metadata as `.mat`: timestamp, `K`, `R_world_cam`, `t_world_cam`, UAV body pose, frame index.
- Model version and random seed for reproducibility.

## 4. Deep-Learning Pipeline

Use semantic segmentation rather than object detection. Crop disease, drought stress, and weeds often occupy irregular patches rather than clean bounding boxes. Segmentation gives pixel-level affected area, which directly supports world projection and treatment-map accumulation. A compact U-Net is the first implementation choice because it is simple, CPU-deployable, ONNX-friendly, and thesis-readable. DeepLabv3+ is a stronger later baseline if GPU training time is available.

Recommended five classes:

| ID MATLAB | ID Python | Class | Affected |
|---:|---:|---|---|
| 1 | 0 | soil | no |
| 2 | 1 | healthy_crop | no |
| 3 | 2 | diseased_crop | yes |
| 4 | 3 | weed_patch | yes |
| 5 | 4 | water_stress | yes |

Synthetic dataset generation:

```text
python deep_learning/generate_synthetic_dataset.py --count 1000 --seed 2026
```

Training:

```text
python deep_learning/train.py --epochs 30 --batch-size 8 --cpu
```

Inference:

```text
python deep_learning/infer.py ^
  --image results/perception_logs/frames/frame_000010.png ^
  --model deep_learning/weights/crop_segmentation.onnx ^
  --output-mask results/perception_logs/masks/frame_000010_mask.png ^
  --output-probs results/perception_logs/masks/frame_000010_probs.npz
```

ONNX export:

```text
python deep_learning/export_onnx.py --opset 17
```

Augmentations relevant to aerial imagery:

- Random horizontal/vertical flips and 90-degree rotations.
- Brightness and contrast changes for sunlight variation.
- Mild motion blur for forward flight.
- Perspective or affine warp for roll/pitch deviations.
- Gaussian noise and JPEG compression for sensor artifacts.

Realistic thesis targets:

- Synthetic validation mIoU: 0.80 to 0.90.
- Affected-class precision: 0.75 to 0.90.
- Affected-class recall: 0.80 to 0.92.
- Real or manually perturbed synthetic mIoU: 0.55 to 0.75, clearly discussed as synthetic-to-real transfer.

## 5. Image-to-World Mapping

For each pixel `p = [u, v, 1]^T`, compute the normalized camera ray:

```text
r_c = normalize(K^-1 p)
r_w = R_wc r_c
X(lambda) = t_wc + lambda r_w
lambda = (z_ground - t_wc,z) / r_w,z
```

The ground point is valid when `lambda > 0` and `abs(r_w,z)` is not near zero. Return `world_xy = [X_x, X_y]`.

MATLAB signature:

```matlab
world_xy = pixel_to_world(pixel_uv, K, R_uav, t_uav, ground_z)
```

Treatment map:

- Field: 15 m x 10 m.
- Grid resolution: 0.25 m.
- Grid size: 60 x 40 cells.
- Store log-odds affected probability, first detection time, last update time, observation count, and sprayed flag.

Overlap handling:

- Recommended: Bayesian/log-odds update.
- Simpler fallback: max-pooling affected probability per cell.
- Avoid only majority vote for early thesis experiments because class imbalance can hide small affected patches.

## 6. Closed-Loop Replanner

Phase 1: run the existing lawnmower coverage trajectory for each controller. The perception loop is passive: it captures frames, runs segmentation, updates the treatment map, and logs first detection timestamps.

Phase 2: after primary coverage, threshold the treatment map and extract connected components. Compute zone centroids and order them with a nearest-neighbor TSP heuristic. For a 15 m x 10 m field, this is sufficient and easy to explain; exact TSP is unnecessary unless many zones are generated.

Phase 3: generate revisit waypoints at the same 4 m spray altitude. During revisit, log a spray event when UAV XY is within `spray_radius_m` of the zone centroid and the dwell time has elapsed.

New metric:

```text
Detection-to-Treatment Latency = spray_time(zone) - first_detection_time(zone)
```

Report mean, median, max, and failure rate for zones that were detected but not treated before simulation end.

## 7. Integration Plan With Existing MATLAB Code

Add a perception config block near the existing simulation configuration:

```matlab
config.perception = perception_config();
config.perception.enabled = true;
config.perception.seed = mc_seed;
```

At the start of each controller/trajectory run:

```matlab
if config.perception.enabled
    rng(config.perception.seed);
    treatment = treatment_map("init", [], config.perception);
    field_model = create_or_load_field_model(config.perception);
end
```

Inside the simulation loop, after dynamics updates the UAV state:

```matlab
if config.perception.enabled && should_capture_frame(t, frame_rate, dt)
    frame = camera_simulator(uav_state, field_model, config.perception, t, frame_idx);
    mask_probs = run_python_onnx_inference(frame, config.perception);
    affected_pixels = select_affected_pixels(mask_probs, config.perception);
    world_xy = pixel_to_world(affected_pixels, frame.K, frame.R_world_cam, frame.t_world_cam, 0);
    treatment = treatment_map("update", treatment, world_xy, mask_probs.probs, t, config.perception);
end
```

After the lawnmower phase:

```matlab
if config.perception.enabled && config.perception.replanner.enabled
    treatment = treatment_map("zones", treatment, config.perception);
    revisit = revisit_planner(trajectory_log, treatment, config.perception);
    % Run the same selected controller on revisit. Log spray events and latency.
end
```

New config fields:

- `config.perception.enabled`
- `config.perception.seed`
- `config.perception.camera.K`
- `config.perception.camera.fps`
- `config.perception.model.onnx_path`
- `config.perception.model.runtime`
- `config.perception.model.confidence_threshold`
- `config.perception.map.resolution_m`
- `config.perception.replanner.enabled`
- `config.perception.replanner.spray_radius_m`

New thesis report entries:

- Per-controller coverage percentage.
- Per-controller affected-zone recall.
- Mean/median/max detection-to-treatment latency.
- False-alarm rate per square meter.
- Perception runtime per frame.

New figures:

- `detection_overlay_<traj>_<ctrl>.png`
- `treatment_map_<traj>_<ctrl>.png`
- `latency_comparison.png`
- Optional: `runtime_perception_<ctrl>.png`

## 8. Code Skeletons

Skeleton files created in this package:

- `perception/camera_simulator.m`
- `perception/render_field.m`
- `deep_learning/train.py`
- `deep_learning/infer.py`
- `deep_learning/export_onnx.py`
- `mapping/pixel_to_world.m`
- `mapping/treatment_map.m`
- `replanner/revisit_planner.m`
- `configs/perception_config.m`
- `tests/test_pixel_to_world.m`

Additional support files:

- `deep_learning/generate_synthetic_dataset.py`
- `deep_learning/requirements.txt`
- `configs/classes.json`
- `docs/projection_math.md`
- `docs/calibration_notes.md`
- `datasets/README.md`

## 9. Evaluation and Thesis Narrative

New research contributions:

1. A unified comparison of five modern quadrotor control strategies under closed-loop agricultural perception.
2. A reproducible synthetic crop-perception benchmark linked to UAV pose, wind, noise, and controller tracking behavior.
3. A treatment-map feedback mechanism connecting semantic segmentation outputs to target revisit planning.
4. New joint autonomy metrics: coverage quality, affected-zone recall, false alarm rate, and detection-to-treatment latency.

Suggested experiments:

| Factor | Levels |
|---|---|
| Controller | PID, LQR, H-infinity, MPC, SMC |
| Perception | off, passive mapping, active revisit |
| Wind | nominal, moderate, strong |
| Trajectory | lawnmower, circular reference stress test |
| Sensor noise | baseline, doubled camera pose noise |

Suggested perception chapter figures and tables:

- Camera geometry diagram and field footprint at 4 m.
- Synthetic dataset examples with ground-truth masks.
- Segmentation validation table: mIoU, affected precision, affected recall.
- Detection overlays for each controller under the same disturbance seed.
- Treatment maps by controller.
- Detection-to-treatment latency bar chart.
- Runtime table: frame capture, inference, projection, map update.

Abstract update:

This thesis extends a comparative study of PID, LQR, H-infinity, MPC, and sliding mode control for agricultural quadrotor UAVs by adding an opt-in perception and decision layer for precision spraying. During simulated lawnmower flight, a downward-facing RGB camera captures synthetic field imagery at spray altitude, a semantic segmentation model identifies healthy crop, affected crop, weeds, water stress, and soil, and detected affected pixels are projected into a world-frame treatment map using calibrated camera geometry and UAV pose. After coverage, affected zones are extracted and revisited using a TSP-style planner while the same controller set tracks the generated trajectory. The resulting evaluation compares both flight-control performance and agricultural task performance through tracking RMS, coverage, affected-zone recall, false-alarm rate, perception runtime, and detection-to-treatment latency under stochastic wind, sensor noise, and plant-condition variation.

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Synthetic-to-real gap | High synthetic accuracy may not transfer to real fields | Report synthetic and perturbed-synthetic separately; add real images when possible; use aggressive photometric augmentation |
| MATLAB-Python bridge latency | Perception could slow simulation | Run perception at 5 Hz; cache frames; profile inference; switch to MATLAB Python engine if needed |
| ONNX compatibility | MATLAB import may fail for some ops | Use ONNX Runtime as primary runtime; export with opset 17; keep model architecture simple |
| Camera sim fidelity | Over-simple imagery weakens thesis | Add row geometry, patch variation, blur, brightness, perspective; document limitations explicitly |
| Class imbalance | Affected patches are small | Use class weighting or focal loss; report affected recall, not only mIoU |
| Pose convention errors | Projection map can be mirrored or shifted | Unit-test center/corner rays; plot camera footprint; validate with known synthetic patch centroids |

## 7-Day Implementation Plan

Day 1: Add `configs/`, `perception/`, `mapping/`, `replanner/`, and `deep_learning/` folders to the MATLAB project root. Run `test_pixel_to_world.m` and verify camera footprint plots over a flat field.

Day 2: Integrate `perception_config.m`, `camera_simulator.m`, and `render_field.m` into one trajectory/controller run with perception enabled. Save frames and pose metadata at 5 Hz.

Day 3: Generate 1000 synthetic training samples. Train the U-Net baseline on CPU or GPU, export ONNX, and validate `infer.py` on saved MATLAB frames.

Day 4: Connect MATLAB to Python inference via `system()`. Convert masks/probabilities to affected pixels, project to world coordinates, and update the treatment map.

Day 5: Implement treatment-zone extraction, revisit planning, and spray-event logging. Compute detection-to-treatment latency for one controller.

Day 6: Run all five controllers with identical seeds for perception-on active revisit. Generate detection overlays, treatment maps, and latency comparison figures.

Day 7: Extend the thesis summary report with perception metrics, runtime profiling, risk discussion, and the updated abstract. Freeze seeds and archive model/config versions.
