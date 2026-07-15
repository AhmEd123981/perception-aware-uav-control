# End-to-End: Drone Crop-Problem Detection (real data)

The complete vision, working on **real** aerial imagery:
**train → classify → new photo tells the problem → in-flight detection + treatment map.**

## Step 1–2 — Train + classify (real data)
- Dataset: **Agriculture-Vision CVPR-2020** (2017+2018+2019), public AWS, no registration.
- Balanced CPU subset: 4,500 / 900 / 900 patches; classes: background, drydown, weed_cluster, double_plant.
- Model: U-Net (ResNet-18), resumable CPU training (`train_av_resume.py`).
- **Held-out test mIoU 0.405** (drydown recall 0.57, weed 0.35, double_plant 0.31).
- Details: `AV_RESULTS.md`.

## Step 3 — New photo → problem report
Give the model any fresh aerial frame; it names the problem + % of field.
```
python deep_learning/av_infer_new.py --image <photo.jpg>
```
Output example: *"PROBLEM DETECTED: weed_cluster, 22% of field → recommend treatment."*
Also `av_field_flyover.py` = a whole-field survey in Python (fly → detect → map → revisit route).

## Step 4 — Inside the MATLAB drone flight simulation
The flight loop now flies over a **real** field and runs the **real** model.

**Wiring (what connects the model to the sim):**
| File | Change |
|---|---|
| `configs/perception_config.m` | onnx_path → `av_segmentation_2020.onnx`; classes → 4 real; `affected_class_ids = [2 3 4]`; `input_size = [256 256]` |
| `deep_learning/infer_worker.py` | `CLASSES` → 4 real classes |
| `main_comparison_with_perception.m` (`build_field_model`) | loads `datasets/real_field/field_texture.png` → `render_field.m` orthomosaic mode |

**Test:**
```
matlab -batch "addpath(genpath(pwd)); day8_real_field_perception"
```
Result (`results/perception_logs/day8_real_field/`): flew **24 frames** over the real field,
real model ran at ~80 ms/frame, detected **411,810 problem pixels**, treatment map extracted
**9 affected zones** → `treatment_probability.png`.

## Honest limitations
- Real-data accuracy (mIoU 0.405) is far below the old synthetic 0.956 — but it is *real* and meaningful; drydown is the strongest class, weed/double_plant are detected but under-covered.
- The demo field is stitched from problem-heavy real patches, so it shows a lot of detections; a healthier field would show fewer.
- Bigger gains need a GPU (full-res, whole dataset) or the 2021 9-class set.

## The whole pipeline in commands
```
python deep_learning/av_build_2020.py         # 1. get real data
python deep_learning/train_av_resume.py --root C:\Users\Ahmed\av_work\av_subset_2020 --tag 2020 --pretrained
python deep_learning/av_infer_new.py --image <photo.jpg>              # 3. new photo -> problem
python deep_learning/av_field_flyover.py                              # 3b. python field survey
matlab -batch "addpath(genpath(pwd)); day8_real_field_perception"    # 4. in-sim flight
```
