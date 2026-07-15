# Calibration Notes

- Camera model: pinhole RGB camera, 1280 x 720 pixels, 60 degree horizontal FOV.
- Intrinsics are derived from FOV for simulation and should be replaced by a
  checkerboard-calibrated `K` matrix for real data.
- Extrinsics assume a rigid downward-facing camera. In the MATLAB pipeline,
  use a single `R_body_cam` matrix and keep the convention documented next to
  the simulator state definition.
- Synchronization requirement: store timestamp, UAV position, body rotation,
  frame index, and model version for every captured frame.
