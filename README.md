# A Perception-Aware Control Framework for Agricultural Quadrotor UAVs

Deep-learning crop-anomaly segmentation and treatment mapping, integrated into a
five-controller (PID, LQR, H∞, MPC, SMC) quadrotor comparison.

**Author:** Ahmed Ashraf Gouda — ITMO University, MSc Mechatronics and Robotics, 2026.

This repository extends a comparative study of five trajectory-tracking controllers
with an opt-in perception-and-decision layer for precision spraying. During a
simulated lawnmower survey, a downward-facing RGB camera captures the field, a U-Net
(ResNet-18 encoder) trained on the real **Agriculture-Vision CVPR-2020** dataset
segments crop anomalies, affected pixels are projected into a world-frame treatment
map, and a nearest-neighbour planner revisits and "sprays" the detected zones. The
five controllers are then compared on *task-level* metrics, not just tracking error.

## Headline results

- Segmentation held-out test mean IoU: **0.405** (honest real-imagery accuracy).
- On the coverage mission, **PID / MPC / SMC** detect every affected zone
  (recall 1.00) at 88–100 % coverage, while **LQR / H∞** miss half (recall 0.50).
- A 10-seed Monte Carlo campaign confirms PID and SMC are the most reliable.
- Full write-up: [`Perception_Report.pdf`](Perception_Report.pdf)
  (source: [`Perception_Report.tex`](Perception_Report.tex)).

## Repository layout

```
controllers/    PID, LQR, H-infinity, MPC, SMC implementations (MATLAB)
dynamics/       quadrotor rigid-body dynamics
trajectories/   lawnmower and circular reference generators
perception/     camera simulator, ONNX worker bridge, perception loop
mapping/        pixel-to-world projection, log-odds treatment map
replanner/      revisit planner and opportunistic spray executor
deep_learning/  U-Net training / inference / ONNX export (PyTorch)
analysis/       metrics and figure generation (MATLAB + Python)
scripts/        day-by-day demos, Monte Carlo, test runner, orchestrator
tests/          MATLAB unit tests (projection, footprint, map, planner, spray)
configs/        perception_config.m, classes.json
docs/           projection math and calibration notes
results/analysis/   summary CSVs and report figures (tracked)
```

## Not stored in git (large / regenerable)

To stay within GitHub limits, the following are excluded (see `.gitignore`):

| Artifact | Size | How to obtain |
|---|---|---|
| `results/` simulation frames & run logs | ~22 GB | Regenerate — see below |
| `datasets/` raw Agriculture-Vision imagery | ~1.5 GB | [Agriculture-Vision CVPR-2020](https://www.agriculture-vision.com/) |
| `deep_learning/weights/*.pt`, `*.onnx` | ~600 MB | Retrain, or export via `export_onnx.py` |

The training/test metric JSONs (`deep_learning/weights/*_2020.json`) *are* tracked,
so the reported accuracy numbers are reproducible without the weight binaries.

## Reproduce

```powershell
# Python deps
pip install -r deep_learning/requirements.txt

# MATLAB unit tests (expect 5/5 files passing)
matlab -batch "addpath(genpath(pwd)); run_all_tests"

# Full pipeline (day8 real-field + day6 lawnmower + circular + analysis)
matlab -batch "addpath(genpath(pwd)); run_thesis_rerun"
# Add the 10-seed Monte Carlo (slow):
matlab -batch "addpath(genpath(pwd)); run_thesis_rerun(10)"

# Refresh figures/tables
python analysis/thesis_analysis.py
```

Requirements: MATLAB (R2024b used here) with Image Processing Toolbox, Python 3.12
with `onnxruntime`, and a CPU is sufficient for inference (~90–150 ms/frame at 5 FPS).

## Model

U-Net with a ResNet-18 encoder, 4 classes (background, drydown, weed_cluster,
double_plant), 256×256 RGB tiles, exported to ONNX for CPU inference inside the
MATLAB flight simulation. Class ids are kept symmetric across Python (0-based) and
MATLAB (1-based); the affected/spray classes are MATLAB ids `[2 3 4]`.
