# Go2 EDU hardware handoff: read-only telemetry first

**Scope of the first hardware session:** telemetry validation and a supervised manual
safety check. **Not in scope:** autonomous control, learned control, low-level
motor commands, or running any code from this repo that sends commands to the
robot. Nothing in this repository talks to robot hardware.

## 1. Confirm on site (record answers in this file)

- [ ] Robot variant: Go2 **EDU** (needed for SDK/DDS access). Serial number.
- [ ] Firmware version (Unitree app → device info) and SDK version on the onboard computer.
- [ ] Onboard compute: Jetson module model, JetPack/L4T version, free disk.
- [ ] Network: wired Ethernet interface and the robot's documented subnet (from the
      official Unitree docs/manual; don't scan the network to discover it).
- [ ] Available sensors and rates: IMU, joint states, foot contacts, front camera,
      depth, LiDAR (L1/L2 model).
- [ ] Battery level and battery-swap procedure.

## 2. Discover state topics with official tooling (read-only)

Use the official `unitree_ros2` repository (https://github.com/unitreerobotics/unitree_ros2)
and follow its README to set up the CycloneDDS environment on the laptop's wired
interface. Then, without publishing anything:

```
ros2 topic list
ros2 topic info <topic> -v
ros2 topic hz <topic>
ros2 topic echo <topic> --once
```

Topic names and message types must come from the actual output, not from guesses.
Record them here, with the measured rates.

## 3. Read-only logging plan

- Record with `ros2 bag record` for: base state / odometry, IMU, joint states, foot
  contacts, camera image + camera_info, LiDAR if present.
- Session structure: (a) robot powered on and lying down, 60 s; (b) standing still
  under the operator's remote, 60 s; (c) operator drives slow straight lines and turns
  with the **official remote/app**, 2–3 min. Log the operator's commands separately
  if the remote's command topic is visible.
- After the session, convert bags to this repo's episode schema with a converter
  (to be written; not in the MVP) and run `python -m go2wm.data.validate`.

## 4. Time sync and calibration checklist

- [ ] Laptop clock synced to the robot (chrony/NTP). Record the measured offset.
- [ ] Every message has a header stamp. Record the bag-receive vs header-stamp skew.
- [ ] Camera intrinsics (camera_info or a checkerboard calibration). Record resolution, FOV and FPS.
- [ ] Camera-to-base extrinsics (from the URDF in `unitree_ros`, confirmed with a tape measure).
- [ ] IMU frame orientation vs the body FLU convention (lift and tilt the nose; check the sign).

## 5. Sim/real schema parity checks

| quantity | sim (this repo) | real (fill in) |
|---|---|---|
| base frame | FLU | |
| quaternion order | wxyz | |
| gravity_b when level | (0, 0, -1) | |
| velocity command frame | body vx, vy, yaw_rate | |
| command rate | 10 Hz | |
| velocity limits used in sim | vx [-0.4, 1.0], vy ±0.4, yaw ±1.0 | |

## 6. Safety procedure (before any motion, including manual)

- Two people: an operator holding the official remote, and a spotter. The spotter
  holds the **emergency stop / damping** action and has rehearsed it once.
- Clear area at least 4 × 4 m, no bystanders, flat floor, no stairs or drop-offs.
- Speed limited to the slowest gait / speed setting on the remote.
- Check battery level, leg range of motion, and that the robot stands up and lies down cleanly.
- Any high-level command test beyond the remote (e.g. a velocity command from the SDK)
  needs a written, approved procedure and is **not** part of this first session.
