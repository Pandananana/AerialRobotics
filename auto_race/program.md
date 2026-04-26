# Race Autoresearch — research playbook

This is the **human-edited** playbook (analogous to Karpathy's `program.md`).
The agent reads this file at the start of each session and follows it. Update
this file (not the agent) when you want to change how experiments are run.

## Goal

Minimize the **mean of `lap2 + lap3`** across the eval suite, subject to no
crash, no missed gates on any lap, and `lap1 < 240s` on every seed.

## Eval suite

Three seeds, defined in `runner.SEEDS`. Chosen to span detection, cornering,
and skip-handling stress:

- `race_skip_gate_0` — primary unsolved env; the new challenge
- `orthogonal_gate_1` — rotated-gate stress (attitude + yaw lookahead matter)
- `skip_gate_3` — gate-skip stress (planner robustness)

Suite cost = mean per-seed cost. Per-seed cost = `lap2 + lap3` if all
constraints pass, else `inf`. Any single seed = inf → suite = inf. The
runner aborts the suite early on the first inf to save time.

## What can be modified

Only these files:

- `controllers/main/assignment/my_assignment.py`
- `controllers/main/exercises/ex1_pid_control.py`

Do NOT modify `main.py`, `tune.py`, `auto_tune/*`, or `auto_race/*` from this
loop. Do NOT change `SEEDS` from this loop.

## The loop

1. `uv run python -m auto_race.race show` — read recent rows + baseline
2. Read the current state of the two scope files; pick **one** change
3. Edit the file(s)
4. `uv run python -m auto_race.race run --note "<one-line description>"`
5. Compare printed `suite_cost` to `baseline` and apply the keep/revert rules
6. Either:
   - `uv run python -m auto_race.race commit -m "<note>"`  (one line, no co-author)
   - `uv run python -m auto_race.race revert`

## Keep / revert criteria

Keep iff **all** of:

- `suite_cost` is finite (no seed crashed/timed out, all gates passed, lap1
  under 240s on every seed)
- `suite_cost` is **strictly** better than the best finite `suite_cost`
  recorded in `results.tsv`
- No individual seed's per-seed cost regressed by more than **1.5s** vs that
  seed's best finite per-seed cost in prior rows (read the per_seed column)

Otherwise revert. There is no partial credit. We optimize the suite, not a
single seed.

## Iteration rules

- **One change per experiment.** One axis at a time. No combined PID +
  trajectory changes in the same run.
- After **3 consecutive reverts** on the same axis (e.g. `VEL_LIM_XY`),
  pause that axis and try a different hypothesis from the list below.
- Prefer multiplicative tweaks (×0.7, ×1.5) early; switch to additive when
  close to the apparent optimum.
- If a change requires a coupled PID change to be safe (e.g. raising
  `VEL_LIM_XY` saturates `L_vel_xy`), do the trajectory limit alone first,
  observe whether it saturates, then in a *separate* experiment raise the
  matching PID limit.
- Always read the latest rows of `results.tsv` first; do not re-try a change
  that has already been logged as a revert.

## High-EV things to try (rough priority — hints, not a plan)

The agent is free to reorder based on evidence. Numbers are starting points.

1. **Trajectory speed/accel limits** in `PolyTrajectory` (top of class):
   `VEL_LIM_XY = 2.0`, `VEL_LIM_Z = 0.75`, `ACC_LIM_XY = 6`, `ACC_LIM_Z = 5`.
   Direct caps on racing speed. Likely the biggest single lever.
2. **PID saturation limits** in `quadrotor_controller.__init__`:
   `L_acc_rp`, `L_vel_xy`, `L_vel_z`. Coupled with (1) — raising trajectory
   speed without raising these wastes the headroom.
3. **`MeasuringState` dwell** (`self.wait_timer >= 1.0`): biggest lap-1 cost.
   Lower bound is "Kalman has converged" — try 0.5s, 0.3s.
4. **Yaw scheduling** in `RacingState` / `PolyTrajectory.yaw_at`: yaw
   currently follows velocity heading. Try lookahead — aim at the **next**
   gate before reaching the current one (smoother turns through the gate).
5. **Racing-line offsets**: `build_racing_trajectory` uses `m['center']` as
   the waypoint per gate. Replace with offsets that cut the apex (shift the
   waypoint inward toward the chord between adjacent gates).
6. **Final velocity**: `PolyTrajectory.__init__` defaults `v_final=None` →
   zero. Non-zero `v_final` for the final waypoint avoids decel-to-stop.
7. **Pass-through tightening**: `MeasuringState` builds `pass_target =
   center + direction_norm * 0.2`; `PassingThroughState` waits for
   `dist < 0.05`. Tightening these can shave time per gate.
8. **`RepositioningState`** distance / target Z (2m outward, 1.75m height):
   heuristics; faster reacquisition = less lap-1 time.
9. **`clamp_control_command`** caps (`max_speed=2`, `max_yaw_rate=0.4`):
   active outside takeoff/racing — may bottleneck the search/approach phase.
10. **`GateDetector`** margin (15px border rejection): may force unnecessary
    re-yaw on partially-occluded gates seen at the edge of the FOV.
11. **PID gains**: only after the trajectory side is exhausted. A controller
    tuned for step response is not necessarily tuned for high-bandwidth
    tracking. The `auto_tune/` cascade results may be reusable as a starting
    point for individual gain experiments here.

## Anti-patterns

- Tuning PID gains while the trajectory-speed cap is the bottleneck.
- Combining a trajectory-limit raise with a PID-limit raise in the same
  experiment — you won't know which mattered.
- Committing a change that improves the mean but crashes one seed.
- Editing `SEEDS` to "fix" a seed that the change broke.
- Editing `main.py` to alter the eval (the runner restores it on each run).
- Adding `Co-Authored-By` lines to commits — omit them.
