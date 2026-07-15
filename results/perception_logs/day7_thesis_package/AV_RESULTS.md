# Real-Data Crop-Problem Detection (Agriculture-Vision)

A deep-learning model that looks at a **real aerial field photo** and segments the
**problem** in the field (dry crop, weeds, double-planted rows). This replaces the
earlier *synthetic* U-Net, which scored mIoU 0.956 on fake images but cannot
generalise to real photos.

## Final model (2020 combined)
- **Source:** Agriculture-Vision, CVPR-2020 release — public AWS bucket
  `s3://intelinair-data-releases/agriculture-vision/cvpr_paper_2020/`, anonymous
  download (no registration).
- **Data:** full 2020 dataset = **2017 + 2018 + 2019** "miniscale" packages
  (real 512×512 aerial RGB patches with per-pixel anomaly labels).
- **Balanced CPU subset:** **4,500 train / 900 val / 900 test**, drawn evenly
  across the three years, split by field (no field leaks across splits).
  Built by `deep_learning/av_build_2020.py` (+ `av_prepare_subset_v2.py`).
- **Classes:** background, drydown (dry crop), weed_cluster (weeds),
  double_plant (double-seeded rows). Masks store final ids {0,1,2,3, 255=ignore}.
- **Model:** U-Net (ResNet-18, ImageNet-pretrained), CPU, 256×256,
  class-balanced cross-entropy, 25 epochs (best @ epoch 24). Trained with a
  **resumable** trainer (`train_av_resume.py`) that checkpoints every epoch, so
  it survived several session restarts.

### Held-out test results — **mean IoU 0.405**

| class | IoU | precision | recall |
|---|---|---|---|
| background | 0.812 | 0.896 | 0.896 |
| drydown | 0.422 | 0.619 | 0.570 |
| weed_cluster | 0.206 | 0.338 | 0.346 |
| double_plant | 0.180 | 0.300 | 0.310 |

All four classes are now learned. drydown is strongest; weed and double_plant are
detected but under-covered (they need still more data / a GPU for full-res).

## Progress across model versions

| model | data | classes | test mIoU | notes |
|---|---|---|---|---|
| synthetic | fake images | 5 | 0.956 | meaningless on real photos |
| v1 | 2017, 1.2k | incl. water | 0.30 | only drydown worked; water absent |
| v2 | 2017, 3.0k | double_plant | ~0.49 val | big jump; cut short at ep17 |
| **2020** | **2017+18+19, 4.5k** | double_plant | **0.405 test** | all 4 classes learned, full held-out test |

(v2's 0.49 is a *validation* number on the easier 2017-only set; the 2020 model's
0.405 is a stricter held-out **test** on harder, more diverse 3-year data.)

## Artifacts
- `deep_learning/weights/av_segmentation_2020_best.pt` — final checkpoint
- `deep_learning/weights/av_segmentation_2020.onnx` — ONNX (opset 17) for the drone pipeline
- `deep_learning/weights/training_log_2020.json`, `test_metrics_2020.json`
- `results/analysis/av_predictions_2020/` — real-photo panels: RGB | prediction | ground truth
  (orange = drydown, red = weed_cluster, green = double_plant)

## Reproduce
```
python deep_learning/av_build_2020.py          # download 2018+2019, build combined subset
python deep_learning/train_av_resume.py --root C:\Users\Ahmed\av_work\av_subset_2020 \
       --tag 2020 --pretrained --epochs 25 --batch-size 8 --image-size 256
python deep_learning/av_predict_overlay.py --root C:\Users\Ahmed\av_work\av_subset_2020 \
       --ckpt deep_learning/weights/av_segmentation_2020_best.pt \
       --out results/analysis/av_predictions_2020 --n 12
```

## How to push higher (next steps)
1. **GPU + full resolution (512) over the whole dataset** — biggest win for weed/double_plant.
2. Longer training / focal loss / more anomaly oversampling.
3. Add the 2021 supervised set (9 classes) for more problem types.
4. The synthetic-to-real caveat is gone (this is real data); the remaining caveat
   is that CPU + subset limits accuracy on the rarer classes.
```
