# aprilcube

![](https://raw.githubusercontent.com/younghyopark/aprilcube/main/assets/printing_process.gif)

Generate 3D-printable fiducial targets with ArUco or AprilTag markers, then detect their 6-DOF pose from a camera. Targets can be simple cubes/cuboids or voxel-composed shapes such as T-shapes, chairs, frames, and stair-step objects.

**aprilcube** is a two-part pipeline:

1. **Generator** - creates a multi-color 3MF file with markers on the target surface, ready for dual-color 3D printing.
2. **Detector** - detects the printed target in a camera image and estimates its full 6-DOF pose.

![Voxel shape gallery](https://raw.githubusercontent.com/younghyopark/aprilcube/main/docs/voxel_shape_gallery.png)

## Technical Report

Read the technical report: [AprilCube: 3D-Printable Fiducial Targets for Reliable 6-DoF Pose Estimation](https://github.com/younghyopark/aprilcube/blob/main/docs/paper.pdf).

If you use AprilCube in research, please cite:

```bibtex
@software{park2026aprilcube,
  title={AprilCube: 3D-Printable Fiducial Targets for Reliable 6-DoF Pose Estimation},
  author={Park, Younghyo and Agrawal, Pulkit},
  year={2026},
  url={https://github.com/younghyopark/aprilcube},
}
```

[![Star History Chart](https://api.star-history.com/chart?repos=younghyopark/aprilcube&type=date&legend=top-left)](https://www.star-history.com/?repos=younghyopark%2Faprilcube&type=date&legend=top-left)

## Installation

```bash
pip install aprilcube
```

Requires Python 3.10+ and installs OpenCV 4.x (`opencv-contrib-python`), `numpy`, and `pyyaml`.

## Basic Usage

### Generate a target

Generate a classic cuboid target directly from the CLI:

```bash
aprilcube generate --grid 1x1x1 --dict 4x4_50 --tag-size 30 -o models/basic_cube
```

Generate a voxel-composed target from a YAML spec:

```bash
aprilcube generate examples/t_shape_target.yaml
```

### Robot end-effector mount

Add the included connector and mounting rod to attach an AprilCube to a robot
end effector, such as those on Franka and Flexiv arms:

```bash
aprilcube generate --tag-size 24 --grid 1x3x3 --dict 4x4_1000 \
  --end-effector-connector --connector-rod-length 50 \
  --connector-rod-radius 10
```

This cube is ready to print in [models/calibration_cube](models/calibration_cube)
([cube.3mf](models/calibration_cube/cube.3mf)). [aiofranka](https://github.com/younghyopark/aiofranka)'s
`aiofranka camera calibrate` holds it on a Franka's flange to locate a fixed camera
relative to the robot.

Open the standalone voxel designer and export a YAML spec:

```bash
aprilcube web
```

Generated model directories contain `cube.3mf`, `config.json`, `thumbnail.png`, and MuJoCo/OBJ visualization assets under `mujoco/`.

### Detect pose

Given a BGR camera frame as a NumPy array:

```python
import aprilcube

det = aprilcube.detector(
    "models/basic_cube/config.json",
    {"fx": 800, "fy": 800, "cx": 320, "cy": 240},
)

result = det.process_frame(frame)

if result["success"]:
    print(result["T"])              # 4x4 camera-frame pose
    print(result["reproj_error"])   # Reprojection error in pixels
```

For detailed CLI options, YAML schemas, Python API notes, visualization, async detection, output formats, and detector internals, see [docs/usage.md](https://github.com/younghyopark/aprilcube/blob/main/docs/usage.md).

## Printing

AprilCube targets are designed for dual-color FDM printing on Bambu Lab printers with AMS or AMS Lite.

1. Open the generated `cube.3mf` in Bambu Studio.
2. Use filament colors: extruder 1 = black, extruder 2 = white. PLA is recommended.
3. Slice and print. The 3MF uses `paint_color` attributes for automatic color assignment.

The latest generated 3MF files include Bambu Studio 2.x project metadata for compatibility with current Bambu Studio releases.

## Resources

- [Detailed usage guide](https://github.com/younghyopark/aprilcube/blob/main/docs/usage.md)
- [Technical report](https://github.com/younghyopark/aprilcube/blob/main/docs/paper.pdf)
- [Voxel example gallery](https://github.com/younghyopark/aprilcube/blob/main/docs/voxel_shape_gallery.png)
- [Changelog](https://github.com/younghyopark/aprilcube/blob/main/CHANGELOG.md)

## License

MIT
