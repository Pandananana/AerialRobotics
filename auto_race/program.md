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
   - `uv run python -m auto_race.race commit -m "<note>"` (one line, no co-author)
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
  observe whether it saturates, then in a _separate_ experiment raise the
  matching PID limit.
- Always read the latest rows of `results.tsv` first; do not re-try a change
  that has already been logged as a revert.

## Anti-patterns

- Tuning PID gains while the trajectory-speed cap is the bottleneck.
- Combining a trajectory-limit raise with a PID-limit raise in the same
  experiment — you won't know which mattered.
- Committing a change that improves the mean but crashes one seed.
- Editing `SEEDS` to "fix" a seed that the change broke.
- Editing `main.py` to alter the eval (the runner restores it on each run).
- Adding `Co-Authored-By` lines to commits — omit them.
