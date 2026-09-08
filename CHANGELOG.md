# Changelog

## 0.3.0 - 2026-09-08

- Add an optional robot end-effector connector and configurable mounting rod
  on the target's +Z surface, with CLI/YAML settings and 3MF/OBJ exports.
- Support borders in half-cell increments across target generation, detector
  corner geometry, and the voxel designer.
- Expose `MarkerDetection`, `detect_markers()`, and
  `process_marker_detections()` to reuse marker detection across targets that
  share a dictionary. Add `store_latest=False` for custom overlay pipelines.
- Extend the webcam example to discover multiple target configurations and
  draw a combined detection overlay.
- Add T-shaped target size variants, including a dense half-cell-border layout.
- Bundle the connector STL in wheels and source distributions, and keep source
  archives focused on code, documentation, examples, and tests.
- Require OpenCV 4.x to preserve compatibility with its marker detection API.
