# auto_race — Karpathy-style autoresearch for end-to-end racing

Tiny scaffolding for automated full-race experiments. Built around the
existing headless `run_race.sh`. Sister to `auto_tune/` (which optimizes the
PID step response); `auto_race/` optimizes lap times directly.

## Files

- `program.md` — human-edited research playbook. The agent reads this.
- `runner.py` — patches `CLI_MODE` and `env_name` in `controllers/main/main.py`,
  runs Webots headless once per seed in `SEEDS`, parses stdout, aggregates
  per-race + suite metrics, appends to `results.tsv`. Always restores
  `main.py` on exit (success, error, KeyboardInterrupt).
- `race.py` — small CLI: `show`, `run`, `seed`, `commit`, `revert`.
- `results.tsv` — append-only experiment log (created on first append).

## One experiment

```sh
# inspect
uv run python -m auto_race.race show

# propose a change, edit my_assignment.py / ex1_pid_control.py,
# then run the eval suite
uv run python -m auto_race.race run --note "raise VEL_LIM_XY 2.0 -> 2.5"

# keep
uv run python -m auto_race.race commit -m "raise VEL_LIM_XY 2.0 -> 2.5"

# or discard
uv run python -m auto_race.race revert
```

The CLI does not auto-decide. The keep/revert rule lives in `program.md`.

## Eval suite

Defined in `runner.SEEDS`:

- `race_skip_gate_0` — primary unsolved env
- `orthogonal_gate_1` — rotated-gate stress
- `skip_gate_3` — gate-skip stress

One `run` = three races. Wall-clock ~3–6 min headless on a typical machine.
The suite aborts after the first seed that produces an inf cost.
