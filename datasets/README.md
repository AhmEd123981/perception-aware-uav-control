# Dataset Layout

```text
datasets/
  synthetic/
    images/      RGB synthetic frames
    masks/       single-channel class-id masks
    metadata/    camera pose, seed, class mapping
  real/
    images/      real UAV or orthomosaic crops
    annotations/ manual masks or polygons
  splits/
    train.txt
    val.txt
    test.txt
```

Python masks use zero-based class IDs. MATLAB semantic masks use one-based
class IDs when indexing probability columns.
