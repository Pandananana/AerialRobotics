"""CLI for running one PID autoresearch experiment.

Examples
--------

Print current gains and tuning level:
    uv run python -m auto_tune.tune show

Run a single tune step-response with a candidate gain change:
    uv run python -m auto_tune.tune step --loop rate_rp --P 1.8 --D 0.12 \
        --note "rate_rp: bump P, hold D"

Run a race with current gains, log the result:
    uv run python -m auto_tune.tune race --note "post-rate_rp freeze"

The CLI does not auto-keep or revert. The human (or the agent driving it)
inspects ``auto_tune/results.tsv`` and either commits or restores the file:

    git checkout -- controllers/main/exercises/ex1_pid_control.py
"""

from __future__ import annotations

import argparse
import sys

from . import runner


def _build_gain_updates(args: argparse.Namespace) -> dict[str, float]:
    updates: dict[str, float] = {}
    for term in ("P", "I", "D"):
        v = getattr(args, term, None)
        if v is not None:
            updates[f"{term}_{args.loop}"] = float(v)
    return updates


def cmd_show(_: argparse.Namespace) -> int:
    print(f"tuning_level: {runner.current_tuning_level()}")
    print("gains:")
    for k, v in runner.current_gains().items():
        print(f"  {k:<14} = {v}")
    return 0


def cmd_step(args: argparse.Namespace) -> int:
    updates = _build_gain_updates(args)
    if updates:
        runner.set_gains(updates)
    runner.set_tuning_level(args.loop)
    print(f"running tune for loop={args.loop} updates={updates} ...", file=sys.stderr)
    result = runner.run_tune(timeout=args.timeout)
    if not result.tuning_level:
        print("ERROR: no tuning result block found in stdout. Last 2KB:", file=sys.stderr)
        print(result.raw_stdout[-2048:], file=sys.stderr)
        runner.append_tune_row(args.loop, updates, result, note=f"PARSE_FAIL {args.note}")
        return 2
    runner.append_tune_row(args.loop, updates, result, note=args.note)
    print(
        f"loop={args.loop} cost={result.cost:.4f} "
        f"rise=({result.rise_time_high_s:.3f},{result.rise_time_low_s:.3f}) "
        f"os=({result.overshoot_high_pct:.1f}%,{result.overshoot_low_pct:.1f}%) "
        f"ss=({result.ss_err_high_pct:.1f}%,{result.ss_err_low_pct:.1f}%)"
    )
    return 0


def cmd_race(args: argparse.Namespace) -> int:
    runner.set_tuning_level("off")
    print("running race ...", file=sys.stderr)
    result = runner.run_race(timeout=args.timeout)
    runner.append_race_row({}, result, note=args.note)
    print(
        f"race cost={result.cost:.4f} laps={result.lap_times} "
        f"gates_ok={result.all_gates_passed} "
        f"crashed={result.crashed} timed_out={result.timed_out}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="auto_tune.tune")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_show = sub.add_parser("show", help="print current gains and tuning_level")
    sp_show.set_defaults(func=cmd_show)

    sp_step = sub.add_parser("step", help="run one step-response experiment")
    sp_step.add_argument("--loop", required=True, choices=runner.LOOPS)
    sp_step.add_argument("--P", type=float, default=None)
    sp_step.add_argument("--I", type=float, default=None)
    sp_step.add_argument("--D", type=float, default=None)
    sp_step.add_argument("--note", default="")
    sp_step.add_argument("--timeout", type=float, default=120.0)
    sp_step.set_defaults(func=cmd_step)

    sp_race = sub.add_parser("race", help="run one full race with current gains")
    sp_race.add_argument("--note", default="")
    sp_race.add_argument("--timeout", type=float, default=300.0)
    sp_race.set_defaults(func=cmd_race)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
