# Chat History and Run Log

Project path:

```text
C:\Users\Ahmed\OneDrive\Desktop\try for 5.5
```

Thesis project:

```text
A Perception-Aware Control Framework for Agricultural Quadrotor UAVs:
Deep-Learning Crop-Anomaly Segmentation and Treatment Mapping
Ahmed Ashraf Gouda, ITMO University, 2026
```

This file records the important decisions, fixes, commands, outputs, and current
state of the perception-extension work. It is not a verbatim UI transcript, but it
preserves the technical content needed to reproduce the work. **It reflects the
current real-data (Agriculture-Vision) pipeline** and supersedes any earlier
synthetic-dataset notes.

> Authoritative source of numbers: `Perception_Report.tex` and the CSV/JSON files
> under `results/` and `deep_learning/weights/`. If this log and the report ever
> disagree, the report and the live CSV/JSON win.

## Original Goal

Extend the existing MATLAB quadrotor controller-comparison project with an opt-in
perception and decision module:

- Simulated downward RGB camera at 4 m spray altitude, 5 FPS.
- Semantic segmentation of real aerial farmland anomalies.
- Pixel-to-world projection using camera intrinsics and UAV pose.
- Treatment map accumulation over the 15 m x 10 m field.
- Revisit planner + spray executor for affected zones.
- Spray-event logging and detection-to-treatment latency.
- Per-controller task metrics for PID, LQR, H-infinity, MPC, and SMC.

The extension had to coexist with the original controller framework and remain opt-in.

## Main Delivered Architecture

```text
perception/
  camera_simulator.m
  render_field.m
  run_perception_loop.m
  onnx_worker.m               (persistent Python ONNX Runtime worker)
  controller_adapter.m

deep_learning/
  train.py
  infer.py
  infer_worker.py
  export_onnx.py
  requirements.txt
  weights/                    (av_segmentation_2020.pt/.onnx + metrics/logs)

mapping/
  pixel_to_world.m
  treatment_map.m

replanner/
  revisit_planner.m
  spray_executor.m            (opportunistic executor)

analysis/
  perception_metrics.m
  thesis_analysis.py

scripts/
  day6_controller_comparison_demo.m     (lawnmower)
  day6_controller_comparison_circular.m (circular)
  day8_real_field_perception.m          (real-field GT detection check)
  perception_monte_carlo.m
  run_thesis_rerun.m                     (orchestrator)
  day7_finalize_thesis_package.py
  run_all_tests.m

tests/
  test_pixel_to_world.m
  test_camera_footprint.m
  test_treatment_map.m
  test_revisit_planner.m
  test_spray_executor.m
```

## Bridge Decision

The MATLAB/Python bridge is a persistent Python ONNX Runtime worker (`onnx_worker.m`
+ `infer_worker.py`).

Reason:

- Avoids per-frame Python startup jitter from repeated `system()` calls.
- Keeps MATLAB responsible for control, dynamics, mapping, and reporting.
- Keeps Python responsible for U-Net inference.
- Works on a normal laptop; CPU inference, with optional GPU if ONNX Runtime GPU is
  installed.

## Deep-Learning Model (REAL DATA — Agriculture-Vision CVPR-2020)

The model is a U-Net with a **ResNet-18 encoder**, trained on the **real
Agriculture-Vision CVPR-2020** dataset (combined 2017/2018/2019 miniscale releases).

Classes (4):

```text
0  background
1  drydown          (affected)
2  weed_cluster     (affected)
3  double_plant     (affected)
```

Input tiles are 256x256 RGB; the model is exported to ONNX for CPU inference inside
the MATLAB flight simulation.

Data split: a balanced subset of ~4500 train / 900 val / 900 test patches. Background
dominates (~82% of pixels), which makes the affected classes intrinsically hard.

### Training

- 25 epochs, Adam, resumable checkpointing.
- Best validation mean IoU ~= 0.447 at epoch 24.

Artifacts (the `_2020` set is the current, kept model):

```text
deep_learning/weights/av_segmentation_2020.pt
deep_learning/weights/av_segmentation_2020.onnx
deep_learning/weights/training_log_2020.json
deep_learning/weights/test_metrics_2020.json
```

### Held-out test accuracy (the honest, real-imagery numbers)

```text
mean IoU = 0.405

class          IoU     precision  recall   affected
background     0.812   0.896      0.896    no
drydown        0.422   0.619      0.570    yes
weed_cluster   0.206   0.338      0.346    yes
double_plant   0.180   0.300      0.310    yes
```

This ~0.405 mIoU is far below the ~0.95 that an earlier **synthetic** prototype
reported on its own procedural generator. That synthetic prototype trained and tested
on the same synthetic process and therefore did not measure field accuracy; it has
been removed from the project. **All accuracy claims in the report and here are the
real Agriculture-Vision numbers above.**

ONNX export command:

```powershell
python deep_learning\export_onnx.py --opset 17 --checkpoint deep_learning\weights\av_segmentation_2020.pt
```

## Perception-to-Action Pipeline

- **Camera / projection:** downward pinhole camera at 4 m, 5 FPS; each detected
  affected pixel back-projected onto the flat field plane z=0 via `r = R K^{-1} x`,
  `s = -p_z / r_z`. Class ids are kept symmetric (Python 0-based, MATLAB 1-based).
- **Treatment map:** clamped log-odds accumulator on 0.25 m cells,
  `l_hit = +0.85`, clamp `[-10, 10]`; a cell joins a treatment zone when
  `P = sigma(l) >= 0.60`. Connected affected cells >= 0.50 m^2 (8 cells) become zones.
- **Revisit + spray:** nearest-neighbour tour from the final survey position; the
  spray executor is **opportunistic** — it services every zone the vehicle passes
  within the spray radius of. This replaced an earlier strictly-sequential executor
  whose zero settling margin left zones unsprayed for controllers that overshoot on
  the circular path. The opportunistic executor services every detected zone for all
  five controllers, so latency is finite for all of them.

## Key Fixes Completed

### 1. Real Agriculture-Vision segmentation replaced the synthetic prototype
The synthetic 5-class TinyUNet/procedural model (inflated ~0.955 mIoU) was retired.
The current model is the ResNet-18 U-Net trained on real Agriculture-Vision imagery
(4 classes, test mIoU 0.405).

### 2. Real, pixel-aligned ground truth
The 15x10 m field texture and its ground-truth class mask are generated from the same
real Agriculture-Vision patches, so every affected pixel in the mask corresponds to a
real anomaly in the imagery the camera sees. All detection metrics are scored against
this real mask. Field affected area ~= 23.1%.

### 3. Zone-overlap recall (robust to merged detections)
A ground-truth affected zone counts as recalled when **any** confident detected cell
overlaps it (connected-component overlap), rather than centroid matching. This avoids
a false zero when adjacent detections merge into one large component.

### 4. missed_zones made recall-consistent
`analysis/perception_metrics.m` now derives `missed_zones` from detection against the
ground truth (`compute_zone_detection` -> `n_missed`), so `recall == 1` implies
`missed_zones == 0`. When no ground truth is available it falls back to the count of
detected-but-unsprayed zones. This fixed the day-8 real-field row that previously
reported `missed_zones = 8` in a detection-only run with no spray executor.

### 5. Opportunistic spray executor
See pipeline above — replaced the strictly-sequential zero-margin executor.

## Current Results (all scored against the real ground truth)

### Real-field detection check (day8)

File: `results/perception_logs/day8_real_field/real_field_metrics.csv`

```text
coverage 95%,  recall 1.00,  false_alarm 0.057 m^-2
twelve affected zones localized (1 drydown + 11 weed clusters)
```

### Day 6 lawnmower (single deterministic run)

File: `results/perception_logs/day6_controller_comparison/day6_controller_metrics.csv`

```text
controller  coverage%  recall  false_alarm_m^-2  latency_s
PID         100.0      1.00    0.137             104.7
LQR          56.4      0.50    0.147             102.5
Hinf         62.4      0.50    0.105              91.1
MPC          88.2      1.00    0.083              96.6
SMC         100.0      1.00    0.123             101.9
```

### Day 6 circular (single deterministic run)

File: `results/perception_logs/day6_circular/day6_controller_metrics.csv`

```text
controller  coverage%  recall  false_alarm_m^-2  latency_s
PID          81.0      1.00    0.061              94.9
LQR          32.1      1.00    0.280             145.9
Hinf         54.3      1.00    0.337             125.9
MPC          99.5      1.00    0.158              98.4
SMC          76.6      1.00    0.073             105.9
```

### Monte Carlo (10 seeds, mean +/- 95% CI)

File: `results/perception_logs/monte_carlo/perception_monte_carlo_aggregate.csv`

```text
trajectory    controller  coverage%       recall
agricultural  PID         100.0 +/- 0.0   1.00 +/- 0.00
agricultural  LQR          55.5 +/- 2.5   0.55 +/- 0.10
agricultural  Hinf         62.4 +/- 0.3   0.50 +/- 0.00
agricultural  MPC          86.6 +/- 0.6   1.00 +/- 0.00
agricultural  SMC         100.0 +/- 0.0   1.00 +/- 0.00
circular      PID          81.8 +/- 0.3   1.00 +/- 0.00
circular      LQR          31.2 +/- 6.6   0.20 +/- 0.22
circular      Hinf         33.9 +/- 11.0  0.75 +/- 0.22
circular      MPC          90.9 +/- 4.3   0.90 +/- 0.13
circular      SMC          70.3 +/- 0.8   1.00 +/- 0.00
```

### Perception runtime

Per-frame ONNX inference ~90-150 ms on CPU, comfortably within the 5 FPS camera
budget. Perception is not the bottleneck; the limiting factor is each controller's
ability to hold the survey path.

## Unit Tests

```powershell
matlab -batch "addpath(genpath(pwd)); run_all_tests"
```

Expected: 5/5 passed (test_pixel_to_world, test_camera_footprint, test_treatment_map,
test_revisit_planner, test_spray_executor).

## Reproduction Order

From PowerShell:

```powershell
cd "C:\Users\Ahmed\OneDrive\Desktop\try for 5.5"
pip install -r deep_learning\requirements.txt
```

Optional GPU PyTorch (training only; inference runs fine on CPU via ONNX):

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

Re-run everything (the orchestrator handles ordering):

```powershell
matlab -batch "addpath(genpath(pwd)); run_thesis_rerun"        % day8 + day6 lawn + day6 circular + python refresh
matlab -batch "addpath(genpath(pwd)); run_thesis_rerun(10)"    % adds the 10-seed Monte Carlo (slow; run overnight)
```

Or individual stages:

```powershell
matlab -batch "addpath(genpath(pwd)); day8_real_field_perception()"
matlab -batch "addpath(genpath(pwd)); day6_controller_comparison_demo('Duration',65,'CameraFps',5.0,'ImageSize',256)"
matlab -batch "addpath(genpath(pwd)); day6_controller_comparison_circular('Duration',65,'CameraFps',5.0,'ImageSize',256)"
matlab -batch "addpath(genpath(pwd)); perception_monte_carlo(10)"
python analysis\thesis_analysis.py
python scripts\day7_finalize_thesis_package.py
```

## Notes for Thesis Writing

- The deep-learning result is a **real-data** benchmark (Agriculture-Vision), reported
  honestly at test mIoU 0.405; do not quote the retired synthetic ~0.95.
- The controller comparison runs through the original controller API, not placeholders.
- Recall uses zone-overlap against the real ground-truth mask (robust to merged
  detections), and `missed_zones` is recall-consistent.
- CPU inference is supported and fast enough (5 FPS). GPU is optional and only helps
  training / a larger future model.
- Main result: high survey coverage does not by itself guarantee treatment success;
  PID/MPC/SMC detect every zone on the coverage mission while LQR/H-infinity miss half,
  and the 10-seed Monte Carlo confirms PID/SMC are the most reliable.
