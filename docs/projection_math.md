# Pixel-to-World Projection Notes

For a pixel `p = [u, v, 1]^T`, the camera-frame ray is:

```text
r_c = normalize(K^-1 p)
```

Given camera-to-world rotation `R_wc` and camera origin `t_wc`, the world ray is:

```text
r_w = R_wc r_c
X(lambda) = t_wc + lambda r_w
```

For a flat agricultural field at `z = ground_z`, solve:

```text
lambda = (ground_z - t_wc,z) / r_w,z
```

The projected ground point is valid only when `lambda > 0` and the ray is not
parallel to the ground plane. The treatment map stores `X_x, X_y` in a fixed
0.25 m grid over the 15 m by 10 m field.
