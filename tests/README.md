# Tests

Detection benchmark and integration tests for aprilcube. Renders the cube at known 6-DOF poses using synthetic images, runs the detector, and compares against ground truth.

## Modes

### Discrete test (default)

Evaluates 30+ systematic viewpoints (face head-on, edges, corners, oblique angles) with per-viewpoint pass/fail based on rotation and translation error thresholds.

```bash
# Default: inline 2x2x2 cube, all viewpoints
python tests/test_detection.py

# Use a pre-generated cube config
python tests/test_detection.py --cube models/2x2x2_30_cube/config.json -v

# Save annotated images for inspection
python tests/test_detection.py --save-images test_output/

# With augmentation (occlusion + motion blur)
python tests/test_detection.py --occlusion 0.2 --blur 7 -v
```

### Video mode

Renders a continuous trajectory and writes an annotated `.mp4` with ground truth overlay, detected pose, and error metrics.

```bash
python tests/test_detection.py --video orbit.mp4 --trajectory orbit
python tests/test_detection.py --video full.mp4 --trajectory full --frames 300
python tests/test_detection.py --video stress.mp4 --trajectory stress -v
```

**Trajectories:** `orbit`, `spiral`, `approach`, `tumble`, `translate`, `full`, `wander`, `shake`, `stress`

## Options

| Arg | Default | Description |
|-----|---------|-------------|
| `--cube` | — | Path to `config.json` (generates inline if omitted) |
| `--grid` | `2x2x2` | Grid for inline generation |
| `--dict` | `4x4_50` | Dictionary for inline generation |
| `--tag-size` | `30` | Tag size in mm for inline generation |
| `--resolution` | `800` | Rendered image resolution |
| `--pixels-per-cell` | `20` | Texture resolution per cell |
| `--rot-threshold` | `5.0` | Max rotation error (degrees) for pass |
| `--trans-threshold` | `2.0` | Max translation error (%) for pass |
| `--video` | — | Output `.mp4` path (enables video mode) |
| `--trajectory` | `orbit` | Trajectory type |
| `--frames` | `300` | Number of video frames |
| `--fps` | `30` | Video FPS |
| `--no-filter` | — | Disable Kalman temporal filter |
| `--occlusion` | `0` | Occlusion fraction (0-1) |
| `--blur` | `0` | Motion blur kernel size in px |
| `--drop-rate` | `0` | Frame stall probability (0-1) |
| `--drop-burst` | `1` | Max consecutive stale frames per stall |
| `-v` | — | Verbose per-viewpoint/frame output |
