# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EPFL MICRO-502 Aerial Robotics course. Program a Crazyflie quadrotor to autonomously fly through a race course of gates as fast as possible in Webots simulation.

## Assignment Requirements

- 5 square pink gates arranged in a circle-like pattern, completed counter-clockwise
- Gate positions/sizes are randomized each run. Circle centre at (4, 4), inner radius 1.5, outer radius 3.5, height 0.7–2.0m, opening size 0.3–0.5m, rotation ±π/6
- Lap 1: gate positions unknown — must detect using computer vision (OpenCV)
- Laps 2–3: gate positions known — fly as fast as possible
- 240 second time limit. Clock starts when leaving takeoff pad, stops on return
- Grading: 3.5 for takeoff, +0.25 per gate in lap 1 (max 4.75), laps 2–3 time-based (max 6.0)

## Running

- Open `worlds/crazyflie_world_assignment.wbt` in Webots — it auto-runs `controllers/main/main.py`
- Set `exp_num = 4` and `control_style = 'path_planner'` in main.py for the assignment
- Dependencies managed with UV (`uv sync`), Python 3.13+

## Architecture

**`controllers/main/main.py`** — Main simulation loop: reads sensors, feeds data to a path planner thread, applies PID control. The path planner thread calls `assignment.get_command(sensor_data, camera_data, dt)` which returns a setpoint `[x, y, z, yaw]`. The PID controller (`exercises/ex1_pid_control.py`) converts setpoints to motor PWM via cascaded position→velocity→attitude→rate PIDs.

**Sensor data dict** passed to assignment code includes: `x/y/z_global`, `roll/pitch/yaw`, `v_x/v_y/v_z`, `v_forward/v_left/v_up`, `range_front/left/back/right/down`, `rate_roll/rate_pitch/rate_yaw`, quaternion (`q_x/q_y/q_z/q_w`).

**Camera** returns BGRA numpy array from the drone's FPV camera.

**Progress tracking** — main.py divides the arena into angular segments around (4,4) and tracks which segment the drone is in. A gate is "passed" when the drone physically enters the gate's bounding box in local frame.

**`controllers/main/assignment/my_assignment.py`** — Assignment code, structured as a state machine (`TakeoffState` → `SearchingState` → `ApproachingState` → `MeasuringState` → `PassingThroughState` → next gate or `DoneState`). Key classes: `GateDetector` (OpenCV pink gate detection + pixel-to-world projection), `GateKalmanFilter` (12D filter for 4 gate corners), `GateTracker` (manages filter state and stored measurements), `DroneState` (typed wrapper around sensor dict). The `MyAssignment` orchestrator runs detection, state transitions, and command clamping.

**Other key files:**

- `controllers/main/exercises/ex1_pid_control.py` — PID controller (tunable)
- `controllers/main/lib/` — Helper libraries (PID class, A\* pathfinding)
