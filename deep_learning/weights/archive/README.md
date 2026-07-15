# Archived (superseded) model weights

These files are **not** used by the current pipeline. They are earlier or intermediate
checkpoints kept only for provenance. The live model is the `av_segmentation_2020`
set in the parent `weights/` folder (test mIoU 0.405 on real Agriculture-Vision).

Nothing in the code references the files below by name (verified by grep), so moving
them here does not affect any run. You can delete this folder to reclaim ~438 MB.

| File | What it is |
|------|------------|
| `av_2020_resume.pt` | mid-training resume checkpoint of the 2020 run (~164 MB) |
| `av_segmentation_best.pt` | earlier real-data checkpoint (pre-final) |
| `av_segmentation_v1_best.pt` | earlier real-data run, v1 |
| `av_segmentation_v2_best.pt` | earlier real-data run, v2 |
| `av_segmentation.pt` | earlier real-data checkpoint |
| `av_segmentation.onnx` | ONNX export of a superseded checkpoint |
| `training_log_av.json` | training log of a superseded run |
| `test_metrics_av.json` | test metrics of a superseded run |
| `test_metrics_av_v1.json` | test metrics of the v1 run |

To restore any file, move it back up one level into `deep_learning/weights/`.
