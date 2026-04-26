# auto_tune — Karpathy-style autoresearch for the PID controller

Tiny scaffolding to run automated PID tuning experiments. Built around the
two existing headless scripts (`run_tune.sh`, `run_race.sh`) and the metrics
already printed by `ex1_pid_control.py:plot()`.

## Files

- `program.md` — the human-edited research playbook. The agent reads this.
- `runner.py` — patches gains/tuning_level inside `ex1_pid_control.py`,
  invokes Webots headless, parses stdout into structured metrics, appends
  rows to `results.tsv`.
- `tune.py` — a small CLI: `show`, `step`, `race`.
- `results.tsv` — append-only log of every experiment.

## One experiment

```sh
# inspect current state
uv run python -m auto_tune.tune show

# propose a new rate_rp candidate, run the step response, log it
uv run python -m auto_tune.tune step --loop rate_rp --P 1.8 --D 0.12 \
    --note "bump P, hold D"

# if cost improved, commit the change to ex1_pid_control.py.
# otherwise, revert:
git checkout -- controllers/main/exercises/ex1_pid_control.py

# once the cascade is tuned, validate with a race
uv run python -m auto_tune.tune race --note "post-cascade baseline"
```

The `step` and `race` commands always write to `results.tsv`. They do not
commit or revert — that decision is intentionally left to the human (or the
agent following `program.md`).
