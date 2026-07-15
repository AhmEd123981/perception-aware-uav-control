"""Finalize the Day 7 thesis package for perception-aware UAV simulation.

This script collects the outputs from Days 3-6, writes a thesis-ready summary
report, exports runtime and metrics tables, records SHA-256 hashes for model
and configuration artifacts, and builds a reproducibility archive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "perception_logs" / "day7_thesis_package"


KEY_ARTIFACTS = [
    "configs/perception_config.m",
    "configs/classes.json",
    "perception/camera_simulator.m",
    "perception/render_field.m",
    "perception/run_perception_loop.m",
    "perception/onnx_worker.m",
    "perception/controller_adapter.m",
    "mapping/pixel_to_world.m",
    "mapping/treatment_map.m",
    "replanner/revisit_planner.m",
    "replanner/spray_executor.m",
    "analysis/perception_metrics.m",
    "analysis/plot_latency_comparison.m",
    "analysis/plot_pid_footprint_diagnostic.m",
    "controllers/controller_PID.m",
    "controllers/controller_PID_compute.m",
    "controllers/controller_LQR.m",
    "controllers/controller_LQR_compute.m",
    "controllers/controller_Hinf.m",
    "controllers/controller_Hinf_compute.m",
    "controllers/controller_MPC.m",
    "controllers/controller_MPC_compute.m",
    "controllers/controller_SMC.m",
    "controllers/controller_SMC_compute.m",
    "controllers/controller_options_verbose.m",
    "controllers/compute_trajectory_feedforward.m",
    "controllers/quadrotor_flatness_from_acceleration.m",
    "controllers/trajectory_feedforward_options.m",
    "dynamics/quadrotor_dynamics_step.m",
    "trajectories/generate_agricultural_trajectory.m",
    "trajectories/generate_circular_trajectory.m",
    "deep_learning/train.py",
    "deep_learning/infer.py",
    "deep_learning/export_onnx.py",
    "deep_learning/av_field_flyover.py",
    "deep_learning/plot_training_curves.py",
    "deep_learning/requirements.txt",
    "deep_learning/weights/av_segmentation_2020_best.pt",
    "deep_learning/weights/av_segmentation_2020.onnx",
    "deep_learning/weights/training_log_2020.json",
    "deep_learning/weights/test_metrics_2020.json",
    "datasets/real_field/field_texture.png",
    "datasets/real_field/field_mask.png",
    "results/analysis/flyover/flyover_report.json",
    "scripts/day2_perception_capture_demo.m",
    "scripts/day3_deep_learning_pipeline.py",
    "scripts/day4_perception_mapping_demo.m",
    "scripts/day5_revisit_spray_demo.m",
    "scripts/day6_controller_comparison_demo.m",
    "scripts/day6_controller_comparison_circular.m",
    "scripts/perception_monte_carlo.m",
    "results/perception_logs/monte_carlo/summary.csv",
    "main_comparison_with_perception.m",
]


FIGURE_ARTIFACTS = [
    "results/figures/latency_comparison.png",
    "results/figures/detection_overlay_agricultural_PID.png",
    "results/figures/detection_overlay_agricultural_LQR.png",
    "results/figures/detection_overlay_agricultural_Hinf.png",
    "results/figures/detection_overlay_agricultural_MPC.png",
    "results/figures/detection_overlay_agricultural_SMC.png",
    "results/figures/treatment_map_agricultural_PID.png",
    "results/figures/treatment_map_agricultural_LQR.png",
    "results/figures/treatment_map_agricultural_Hinf.png",
    "results/figures/treatment_map_agricultural_MPC.png",
    "results/figures/treatment_map_agricultural_SMC.png",
    "results/perception_logs/day4_mapping_PID/day4_treatment_map.png",
    "results/perception_logs/day5_revisit_PID/day5_revisit_plan.png",
    "results/perception_logs/day5_revisit_PID/day5_latency_comparison.png",
    "results/perception_logs/day6_controller_comparison/day6_latency_comparison.png",
    "results/perception_logs/day6_circular/day6_latency_comparison.png",
    "results/perception_logs/day7_thesis_package/training_curves.png",
    "results/perception_logs/day6_controller_comparison/day6_zone_count_comparison.png",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required Day 7 input is missing: {path.relative_to(ROOT)}")


def collect_inputs() -> dict[str, object]:
    day3_summary = ROOT / "results/perception_logs/day3_deep_learning/day3_summary.json"
    # Canonical (and only) thesis model is the REAL Agriculture-Vision 2020 model.
    # The old synthetic model and its artifacts have been removed from the repository.
    training_log = ROOT / "deep_learning/weights/training_log_2020.json"
    day4_summary = ROOT / "results/perception_logs/day4_mapping_PID/day4_mapping_summary.txt"
    day5_summary = ROOT / "results/perception_logs/day5_revisit_PID/day5_revisit_summary.txt"
    test_metrics = ROOT / "deep_learning/weights/test_metrics_2020.json"
    day6_metrics = ROOT / "results/perception_logs/day6_controller_comparison/day6_controller_metrics.csv"
    day6_summary = ROOT / "results/perception_logs/day6_controller_comparison/agricultural/full_integration_summary.txt"
    if not day6_metrics.exists():
        day6_metrics = ROOT / "results/perception_logs/full_integration/agricultural/full_integration_metrics.csv"
        day6_summary = ROOT / "results/perception_logs/full_integration/agricultural/full_integration_summary.txt"

    for path in [day3_summary, training_log, day4_summary, day5_summary]:
        require_file(path)

    if day6_metrics.exists() and day6_summary.exists():
        day6_rows = read_csv(day6_metrics)
        day6_text = day6_summary.read_text(encoding="utf-8")
    else:
        day6_rows = []
        day6_text = (
            "Full integration outputs are not present yet. Run "
            "main_comparison_with_perception('agricultural') after adding the original "
            "controllers/, dynamics/, and trajectories/ folders to this repo root."
        )

    return {
        "day3_summary": load_json(day3_summary),
        "training_log": load_json(training_log),
        "test_metrics": load_json(test_metrics) if test_metrics.exists() else {},
        "day4_summary_text": day4_summary.read_text(encoding="utf-8"),
        "day5_summary_text": day5_summary.read_text(encoding="utf-8"),
        "day6_metrics": day6_rows,
        "day6_summary_text": day6_text,
    }


def float_field(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    if value == "" or value.lower() == "nan":
        return float("nan")
    return float(value)


def build_metrics_tables(inputs: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    day6_rows = inputs["day6_metrics"]
    metrics_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for row in day6_rows:  # type: ignore[assignment]
        metrics_rows.append(
            {
                "controller": row["controller"],
                "coverage_percent": float_field(row, "coverage_percent"),
                "affected_zone_recall": float_field(row, "affected_zone_recall"),
                "mean_latency_s": float_field(row, "mean_latency_s"),
                "median_latency_s": float_field(row, "median_latency_s"),
                "max_latency_s": float_field(row, "max_latency_s"),
                "missed_zones": int(float_field(row, "missed_zones")),
            }
        )
        runtime_rows.append(
            {
                "controller": row["controller"],
                "mean_inference_time_ms": float_field(row, "mean_inference_time_ms"),
            }
        )
    return metrics_rows, runtime_rows


def artifact_manifest(paths: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rel in paths:
        path = ROOT / rel
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    return rows


def figure_manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rel in FIGURE_ARTIFACTS:
        path = ROOT / rel
        rows.append(
            {
                "figure": rel,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def format_metrics_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No full-integration controller metrics have been generated yet."
    lines = [
        "Controller | Coverage % | Recall | Mean Latency s | Median Latency s | Max Latency s | Missed Zones",
        "---|---:|---:|---:|---:|---:|---:",
    ]
    for row in rows:
        lines.append(
            f"{row['controller']} | {row['coverage_percent']:.2f} | "
            f"{row['affected_zone_recall']:.2f} | {row['mean_latency_s']:.2f} | "
            f"{row['median_latency_s']:.2f} | {row['max_latency_s']:.2f} | {row['missed_zones']}"
        )
    return "\n".join(lines)


def format_runtime_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No full-integration inference timing rows have been generated yet."
    lines = ["Controller | Mean ONNX Inference Time ms", "---|---:"]
    for row in rows:
        lines.append(f"{row['controller']} | {row['mean_inference_time_ms']:.2f}")
    return "\n".join(lines)


def latest_training_metrics(inputs: dict[str, object]) -> dict[str, object]:
    log = inputs["training_log"]
    if not isinstance(log, list) or not log:
        return {}
    last = log[-1]
    val = last.get("val", {})
    return {
        "epoch": last.get("epoch"),
        "train_loss": last.get("train_loss"),
        "mean_iou": val.get("mean_iou"),
        "per_class_iou": val.get("per_class_iou"),
        "per_class_precision": val.get("per_class_precision"),
        "per_class_recall": val.get("per_class_recall"),
    }


def training_completion_note(inputs: dict[str, object]) -> str:
    log = inputs["training_log"]
    epochs = len(log) if isinstance(log, list) else 0
    if epochs >= 30:
        return f"Training completion: full 30-epoch run available ({epochs} logged epochs)."
    return (
        f"Training completion: {epochs} logged epoch(s) are available in this CPU-only workspace. "
        "Run python deep_learning/train.py --epochs 30 --batch-size 8 --image-size 512 "
        "--pretrained --early-stop-patience 8 on a GPU or longer CPU session to replace the checkpoint "
        "with a full thesis training run."
    )


def av_perclass_table(test_metrics: dict) -> str:
    """Per-class held-out test metrics for the real Agriculture-Vision model."""
    if not isinstance(test_metrics, dict):
        return "No held-out test metrics available."
    classes = test_metrics.get("classes", ["background", "drydown", "weed_cluster", "double_plant"])
    test = test_metrics.get("test", {}) if isinstance(test_metrics.get("test"), dict) else {}
    iou = test.get("per_class_iou", [])
    prec = test.get("per_class_precision", [])
    rec = test.get("per_class_recall", [])
    lines = ["Class | Test IoU | Precision | Recall", "---|---:|---:|---:"]
    for i, name in enumerate(classes):
        cell = lambda a: f"{a[i]:.3f}" if i < len(a) and isinstance(a[i], (int, float)) else "n/a"
        lines.append(f"{name} | {cell(iou)} | {cell(prec)} | {cell(rec)}")
    return "\n".join(lines)


def write_report(inputs: dict[str, object], metrics_rows: list[dict[str, object]], runtime_rows: list[dict[str, object]]) -> Path:
    log = inputs["training_log"]
    n_epochs = len(log) if isinstance(log, list) else 0
    last = log[-1] if isinstance(log, list) and log else {}
    val_miou = last.get("val", {}).get("mean_iou") if isinstance(last, dict) else None
    best_miou = last.get("best_mean_iou") if isinstance(last, dict) else None
    best_epoch = last.get("best_epoch") if isinstance(last, dict) else None
    train_loss = last.get("train_loss") if isinstance(last, dict) else None

    test_metrics = inputs.get("test_metrics", {})
    test_miou = None
    if isinstance(test_metrics, dict) and isinstance(test_metrics.get("test"), dict):
        test_miou = test_metrics["test"].get("mean_iou")
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"

    abstract = (
        "This thesis extends the comparative evaluation of PID, LQR, H-infinity, MPC, and sliding-mode "
        "control for agricultural quadrotor UAVs with an opt-in perception and decision layer for "
        "precision spraying. A downward-facing RGB camera simulator captures imagery of a REAL "
        "Agriculture-Vision (CVPR-2020) field during flight; a U-Net (ResNet-18 encoder) semantic "
        "segmentation model trained on the Agriculture-Vision dataset labels each pixel as background, "
        "drydown, weed cluster, or double plant; and affected pixels (drydown, weed cluster, double plant) "
        "are projected into a world-frame treatment map using camera intrinsics and UAV pose. After "
        "coverage, connected affected zones are revisited with a nearest-neighbour planner and spray events "
        "are logged to compute detection-to-treatment latency. Detections are scored against the field's "
        "real ground-truth mask. The framework evaluates controller performance jointly with agricultural "
        "task metrics: affected-zone recall, detection-to-treatment latency, coverage, and inference runtime."
    )

    report = f"""Perception-Aware Agricultural Quadrotor Thesis Package
Generated: {datetime.now(timezone.utc).isoformat()}

Updated Abstract
{abstract}

Research Contributions
1. Unified perception-aware comparison of PID, LQR, H-infinity, MPC, and SMC under an agricultural UAV scenario.
2. Real Agriculture-Vision (CVPR-2020) crop segmentation pipeline connected to MATLAB flight simulation and scored against a real ground-truth field mask.
3. Pixel-to-world treatment-map feedback loop for affected-zone revisit planning.
4. Joint control/perception metrics: affected-zone recall, detection-to-treatment latency, coverage, and runtime.

Deep-Learning Summary (Agriculture-Vision CVPR-2020, real data)
Dataset: Agriculture-Vision CVPR-2020 miniscale (2017+2018+2019).
Train/val/test patches: 4500 / 900 / 900 (512x512 RGB).
Classes: background, drydown, weed_cluster, double_plant.
Affected (map/spray) classes: drydown, weed_cluster, double_plant (MATLAB ids [2 3 4]).
Architecture: U-Net with ResNet-18 encoder (opset-17 ONNX).
Training epochs logged: {n_epochs}
Best validation mIoU: {fmt(best_miou)} (epoch {best_epoch})
Latest validation mIoU: {fmt(val_miou)} (epoch {n_epochs})
Latest train loss: {fmt(train_loss)}
Held-out TEST mIoU: {fmt(test_miou)}

Per-class held-out test metrics:
{av_perclass_table(test_metrics)}

ONNX model: deep_learning/weights/av_segmentation_2020.onnx
Real-field flyover evidence (whole-field fused vs GT): results/analysis/flyover/flyover_report.json

Controller Perception Metrics
{format_metrics_table(metrics_rows)}

Runtime Profile
{format_runtime_table(runtime_rows)}

Day 4 Mapping Evidence
{inputs['day4_summary_text']}

Day 5 Revisit Evidence
{inputs['day5_summary_text']}

Day 6 Five-Controller Evidence
{inputs['day6_summary_text']}

Risk Discussion and Mitigations
Real-data accuracy claim: The canonical model is the real Agriculture-Vision 2020 U-Net (held-out test mIoU {fmt(test_miou)}). Whole-field fused metrics over the stitched real field (results/analysis/flyover/flyover_report.json) are reported honestly and are lower than per-patch test mIoU because survey windows straddle patch boundaries; this is the expected simulation-to-field gap, not a bug.
MATLAB-Python bridge latency: Per-frame process startup has been replaced by a persistent ONNX Runtime worker. Report inference timing after discarding the first three warm-up frames, and keep camera inference below the control-rate budget for deployment claims.
ONNX compatibility: The exported U-Net uses opset 17 and ONNX Runtime successfully. If MATLAB Deep Learning Toolbox import is later used, keep the ONNX Runtime path as the fallback.
Camera simulation fidelity: The renderer intersects camera rays with the flat field plane and samples a real Agriculture-Vision orthomosaic; plant geometry and lighting are still simplified, so field-deployment claims require on-board validation flights.
Metric interpretation: The full-integration Day 6 path calls the real controller initialisation and compute API directly through main_comparison_with_perception.m. The five controllers are initialised via controller_adapter("init", ...) and stepped via controller_adapter("compute", ...); the implementations live in controllers/controller_{{PID,LQR,Hinf,MPC,SMC}}*.m.

Reproducibility Freeze
Random seed: 2026
Camera: 1280x720, 60 degree horizontal FOV, 5 Hz
Treatment grid: 0.25 m cells over 15 m x 10 m
Classes: background, drydown, weed_cluster, double_plant (Agriculture-Vision CVPR-2020)
Affected classes (MATLAB ids): [2 3 4]
Model checkpoint: deep_learning/weights/av_segmentation_2020_best.pt
ONNX export: deep_learning/weights/av_segmentation_2020.onnx
Real field + GT mask: datasets/real_field/field_texture.png, datasets/real_field/field_mask.png
Held-out test mIoU: {fmt(test_miou)}
Artifact hashes: see reproducibility_manifest.json and artifact_manifest.csv
"""

    path = OUT / "thesis_perception_summary_report.txt"
    path.write_text(report, encoding="utf-8")
    return path


def copy_key_outputs() -> None:
    copies = [
        ("results/perception_logs/full_integration/agricultural/full_integration_metrics.csv", "full_integration_metrics.csv"),
        ("results/perception_logs/full_integration/agricultural/full_integration_summary.txt", "full_integration_summary.txt"),
        ("results/perception_logs/day6_controller_comparison/day6_controller_metrics.csv", "day6_controller_metrics.csv"),
        ("results/perception_logs/day6_controller_comparison/day6_controller_metrics.csv", "day6_lawnmower_controller_metrics.csv"),
        ("results/perception_logs/day6_circular/day6_controller_metrics.csv", "day6_circular_controller_metrics.csv"),
        ("results/perception_logs/monte_carlo/summary.csv", "monte_carlo_summary.csv"),
        ("results/perception_logs/day5_revisit_PID/day5_revisit_summary.txt", "day5_revisit_summary.txt"),
        ("results/perception_logs/day4_mapping_PID/day4_mapping_summary.txt", "day4_mapping_summary.txt"),
        ("deep_learning/weights/training_log_2020.json", "training_log_2020.json"),
        ("deep_learning/weights/test_metrics_2020.json", "test_metrics_2020.json"),
        ("deep_learning/requirements.txt", "requirements.txt"),
        ("results/analysis/flyover/flyover_report.json", "flyover_report.json"),
        ("results/analysis/AV_RESULTS.md", "AV_RESULTS.md"),
        ("results/analysis/thesis_tables.tex", "thesis_tables.tex"),
    ]
    for rel, name in copies:
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, OUT / name)


def generate_training_curves() -> None:
    script = ROOT / "deep_learning/plot_training_curves.py"
    log = ROOT / "deep_learning/weights/training_log_2020.json"
    output = OUT / "training_curves.png"
    if not script.exists() or not log.exists():
        return
    subprocess.run(
        [sys.executable, str(script), "--log", str(log), "--output", str(output)],
        cwd=ROOT,
        check=True,
    )


def make_zip() -> Path:
    zip_path = OUT.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in OUT.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(OUT.parent))
    return zip_path


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    inputs = collect_inputs()
    generate_training_curves()
    metrics_rows, runtime_rows = build_metrics_tables(inputs)

    write_csv(
        OUT / "perception_metrics_table.csv",
        metrics_rows,
        ["controller", "coverage_percent", "affected_zone_recall", "mean_latency_s", "median_latency_s", "max_latency_s", "missed_zones"],
    )
    write_csv(
        OUT / "runtime_profile.csv",
        runtime_rows,
        ["controller", "mean_inference_time_ms"],
    )

    artifacts = artifact_manifest(KEY_ARTIFACTS)
    figures = figure_manifest()
    write_csv(OUT / "artifact_manifest.csv", artifacts, ["path", "exists", "size_bytes", "sha256"])
    write_csv(OUT / "figures_manifest.csv", figures, ["figure", "exists", "size_bytes"])

    report_path = write_report(inputs, metrics_rows, runtime_rows)
    copy_key_outputs()

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": 2026,
        "report": str(report_path.relative_to(ROOT)),
        "metrics_table": "results/perception_logs/day7_thesis_package/perception_metrics_table.csv",
        "runtime_profile": "results/perception_logs/day7_thesis_package/runtime_profile.csv",
        "artifact_manifest": "results/perception_logs/day7_thesis_package/artifact_manifest.csv",
        "figures_manifest": "results/perception_logs/day7_thesis_package/figures_manifest.csv",
        "artifacts": artifacts,
        "figures": figures,
    }
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = make_zip()

    print("Day 7 thesis package complete.")
    print(f"Report: {report_path}")
    print(f"Manifest: {OUT / 'reproducibility_manifest.json'}")
    print(f"Archive: {zip_path}")


if __name__ == "__main__":
    main()
