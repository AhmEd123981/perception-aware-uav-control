# Real UAV Imagery

This folder is reserved for real or orthomosaic-sourced agricultural UAV imagery
used to evaluate the synthetic-to-real gap of the segmentation model.

## Layout

```
real/
  images/        Real UAV frames or orthomosaic crops (PNG, 512x512 recommended)
  annotations/   Manual class-id masks matching each image filename
```

## Class IDs

Use the same class IDs as `configs/classes.json`:

| Python | MATLAB | Class           | Affected |
|-------:|-------:|-----------------|----------|
|      0 |      1 | soil            | no       |
|      1 |      2 | healthy_crop    | no       |
|      2 |      3 | diseased_crop   | yes      |
|      3 |      4 | weed_patch      | yes      |
|      4 |      5 | water_stress    | yes      |

## Filename convention

```
images/real_000000.png
annotations/real_000000.png
```

## Why this folder is empty

The current thesis run uses synthetic data only (1000 procedurally rendered
samples in `datasets/synthetic/`). The synthetic-to-real gap is acknowledged in
`THESIS_PERCEPTION_EXTENSION_PLAN.md` Section 10. To populate this folder:

1. Drop real UAV PNGs into `images/`.
2. Annotate masks with the IDs above and save into `annotations/`.
3. Add IDs to a new split file `datasets/splits/real_test.txt`.
4. Re-run `python deep_learning/train.py --eval-split real_test` (or equivalent)
   to report perturbed/real metrics alongside the synthetic results.
