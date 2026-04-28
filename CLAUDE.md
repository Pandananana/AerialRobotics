# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EPFL MICRO-502 Aerial Robotics course. Program a Crazyflie quadrotor to autonomously fly through a race course of gates as fast as possible in Webots simulation.

## Assignment Requirements

- Only modifications in `controllers/main/assignment/my_assignment.py` and `controllers/main/exercises/ex1_pid_control.py` are submitted and evaluated on. Other files will be ignored.
- 5 square pink gates arranged in a circle-like pattern, completed counter-clockwise
- Gate positions/sizes are randomized each run. Circle centre at (4, 4), inner radius 1.5, outer radius 3.5, height 0.7–2.0m, opening size 0.3–0.5m, rotation ±π/6
- Lap 1: gate positions unknown — must detect using computer vision (OpenCV)
- Laps 2–3: gate positions known — fly as fast as possible
- 240 second time limit. Clock starts when leaving takeoff pad, stops on return
- Grading: 3.5 for takeoff, +0.25 per gate in lap 1 (max 4.75), laps 2–3 time-based (max 6.0)

## Running

- Open `worlds/crazyflie_world_assignment.wbt` in Webots — it auto-runs `controllers/main/main.py`
- Set `exp_num = 4` and `control_style = 'path_planner'` in main.py for the assignment
- The PID-tuning exercise world is `worlds/crazyflie_world_excercise.wbt`, which auto-runs `controllers/main/tune.py` (a copy of main.py with `exp_num = 1`)
- Dependencies managed with UV (`uv sync`), Python 3.13+

### Headless runs (for automation / LLM use)

Both scripts launch Webots with `--mode=fast --no-rendering --minimize --batch --stdout --stderr`, so controller `print(...)` output streams to the terminal and there is no GUI interaction. Override the Webots binary path via the `WEBOTS` env var if needed.

- `./run_race.sh` — runs the racing assignment world. After lap 3, `main.py` calls `simulationQuit(0)` and Webots exits.
- `./run_tune.sh` — runs the PID-tuning exercise world. `ex1_pid_control.py` prints step-response metrics (steady-state error, overshoot, rise time) to stdout. Pick which loop to tune by editing `self.tuning_level` in `controllers/main/exercises/ex1_pid_control.py` (`vel_z`, `pos_z`, `vel_xy`, `pos_xy`, `att_rp`, `att_y`, `rate_rp`, `rate_y`, or `off`).
- `uv run python -m auto_race.race run-all -j 4` — sweep every environment in `main.py`'s `environments` dict in parallel and print a per-env table of lap times and gate progress. Use `-j 4` for concurrency.

## Architecture

**`controllers/main/main.py`** — Main simulation loop: reads sensors, feeds data to a path planner thread, applies PID control. The path planner thread calls `assignment.get_command(sensor_data, camera_data, dt)` which returns a setpoint `[x, y, z, yaw]`. The PID controller (`exercises/ex1_pid_control.py`) converts setpoints to motor PWM via cascaded position→velocity→attitude→rate PIDs.

**Sensor data dict** passed to assignment code includes: `x/y/z_global`, `roll/pitch/yaw`, `v_x/v_y/v_z`, `v_forward/v_left/v_up`, `range_front/left/back/right/down`, `rate_roll/rate_pitch/rate_yaw`, quaternion (`q_x/q_y/q_z/q_w`).

**Camera** returns BGRA numpy array from the drone's FPV camera.

**Progress tracking** — main.py divides the arena into angular segments around (4,4) and tracks which segment the drone is in. A gate is "passed" when the drone physically enters the gate's bounding box in local frame.

**`controllers/main/assignment/my_assignment.py`** — Assignment controller built as a state machine: it uses OpenCV gate detection plus Kalman-filtered gate tracking to find and measure the 5 gates on lap 1, then builds a minimum-jerk racing trajectory for the remaining laps. Key classes include `GateDetector`, `GateKalmanFilter`, `GateTracker`, `PolyTrajectory`, `DroneState`, and the `MyAssignment` orchestrator.

**Other key files:**

- `controllers/main/exercises/ex1_pid_control.py` — PID controller (tunable)
- `controllers/main/lib/` — Helper libraries (PID class, A\* pathfinding)
