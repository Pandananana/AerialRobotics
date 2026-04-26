# PID Autoresearch — research playbook

This is the **human-edited** playbook (analogous to Karpathy's `program.md`). The
agent reads this file at the start of each session and follows it. Update this
file (not the agent) when you want to change how experiments are run.

## Goal

Minimize the total lap time of laps 2 and 3 of the racing assignment, subject
to no crash, no missed gates and lap 1 finishing without timeout.

## What can be modified

- The `gains` dict in `quadrotor_controller.__init__`
- The `self.limits` dict in `quadrotor_controller.__init__`

Do not modify anything else in `ex1_pid_control.py`. Do not modify `main.py`,
`tune.py`, or `my_assignment.py` from this loop.

## Cascade order (strict)

Tune from the innermost loop outward. Freeze a loop before moving on.

1. `rate_rp` — body-rate roll/pitch
2. `rate_y` — body-rate yaw
3. `att_rp` — attitude roll/pitch
4. `att_y` — attitude yaw
5. `vel_xy` — horizontal velocity
6. `vel_z` — vertical velocity
7. `pos_xy` — horizontal position
8. `pos_z` — vertical position

After each freeze, re-run the previous loop's step response once as a sanity
check (frozen loops should not regress; if they do, something coupled).

## Per-loop step-response cost

For each step run via `./run_tune.sh` we parse:

- `rise_time_high_s`, `rise_time_low_s`
- `overshoot_high_pct`, `overshoot_low_pct`
- `steady_state_error_high_pct`, `steady_state_error_low_pct`

Cost (lower is better):

```
cost = mean_rise_time + 0.02 * mean_overshoot_pct + 0.05 * mean_ss_err_pct
```

If any rise-time is NaN (never settled / oscillating), cost is treated as
infinite and the candidate is discarded.

## Race cost

For `./run_race.sh` we parse `Lap times: [t1, t2, t3]`, gates progress (All True) and crash status:

```
race_cost = t2 + t3                       (if no crash and t1 < 240)
race_cost = inf                           (if crashed)
race_cost = inf                           (if any lap reported as 1000)
race_cost = inf                           (if any gate is False)
```

Each race candidate must be evaluated on the **same fixed seed**

## Iteration rules

1. Always read the current gains and the latest row of `results.tsv` first.
2. Propose one change per experiment (one or two gains in one loop). Do not
   multi-task across loops.
3. After each run, append a row to `results.tsv`.
4. Keep the change (commit on the working branch) only if cost strictly
   improves and the diagnostic flags are clean (no NaN, no crash).
5. Otherwise `git checkout -- controllers/main/exercises/ex1_pid_control.py`.
6. If 5 consecutive proposals fail, halve the step size of the current
   parameter sweep before continuing.
7. Stop tuning a loop when 10 consecutive proposals fail at the smallest step
   size. Move to the next loop in the cascade.

## Style of changes

- Prefer multiplicative tweaks (×0.7, ×1.5) over additive ones until close to
  optimum.
- Increase `D` only after `P` is near the stability edge.
- Keep `I` at 0 unless a steady-state error >5% persists after `P/D` are tuned.

## Anti-patterns

- Editing limits to "improve" lap time without checking saturation in the
  inner loops first.
- Tuning an outer loop while an inner loop has rise-time NaN.
- Committing a candidate that improves race time but crashed on one of the
  seeds.
