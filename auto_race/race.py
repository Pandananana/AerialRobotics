"""CLI for one race-suite experiment.

Examples
--------

Print eval suite, baseline, and the last few rows:
    uv run python -m auto_race.race show

Run the full 3-seed eval suite at the current code state, log the result:
    uv run python -m auto_race.race run --note "raise VEL_LIM_XY 2.0 -> 2.5"

Run a single seed (debug / smoke test, not logged):
    uv run python -m auto_race.race seed --name race_skip_gate_0

After a successful experiment, commit the in-scope source files:
    uv run python -m auto_race.race commit -m "raise VEL_LIM_XY 2.0 -> 2.5"

After a failed experiment, revert them:
    uv run python -m auto_race.race revert

The keep/revert decision is the caller's, not the CLI's — see ``program.md``.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys

from . import runner


def _fmt(x: float) -> str:
    return "inf" if not math.isfinite(x) else f"{x:.4f}"


def cmd_show(_: argparse.Namespace) -> int:
    print(f"seeds: {', '.join(runner.SEEDS)}")
    print(f"best known suite_cost: {_fmt(runner.best_known_suite_cost())}")
    print("best known per-seed:")
    for seed, cost in runner.best_known_per_seed().items():
        print(f"  {seed:<24} = {_fmt(cost)}")
    if runner.RESULTS_TSV.exists():
        lines = runner.RESULTS_TSV.read_text().splitlines()
        tail = lines[-5:]
        print("recent results:")
        for ln in tail:
            print("  " + ln)
    else:
        print("(no results.tsv yet)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    suite = runner.run_suite(timeout_per_seed=args.timeout)
    runner.append_suite_row(suite, note=args.note)
    print(
        f"suite_cost={_fmt(suite.cost)} worst={_fmt(suite.worst_cost)} "
        f"baseline={_fmt(runner.best_known_suite_cost())}"
    )
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    if args.name not in runner.SEEDS:
        print(
            f"warning: '{args.name}' not in eval suite ({runner.SEEDS})",
            file=sys.stderr,
        )
    r = runner.run_race_for_seed(args.name, timeout=args.timeout)
    print(
        f"seed={args.name} cost={_fmt(r.cost)} laps={r.lap_times} "
        f"gates_ok={r.all_gates_passed} crashed={r.crashed} timed_out={r.timed_out}"
    )
    return 0


def _race_status(r: "runner.RaceResult") -> str:
    if r.crashed:
        return "CRASH"
    if r.timed_out:
        return "TIMEOUT"
    if len(r.lap_times) == 3 and r.all_gates_passed:
        return "OK"
    return "MISS"


def _laps_str(r: "runner.RaceResult") -> str:
    return "[" + ", ".join(f"{t:.2f}" for t in r.lap_times) + "]"


def cmd_run_all(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    names = args.names if args.names else runner.all_env_names()
    width = max(len(n) for n in names)
    workers = max(1, args.parallel)
    print(
        f"running {len(names)} environments "
        f"(timeout={args.timeout:.0f}s each, parallel={workers})\n",
        flush=True,
    )

    results: dict[str, "runner.RaceResult"] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(runner.run_race_for_seed, n, args.timeout): n for n in names
        }
        done = 0
        try:
            for fut in as_completed(futures):
                name = futures[fut]
                done += 1
                r = fut.result()
                results[name] = r
                print(
                    f"[{done}/{len(names)}] {name:<{width}}  "
                    f"{_race_status(r):<7}  laps={_laps_str(r)}  "
                    f"gates={r.gate_progress}",
                    flush=True,
                )
        except KeyboardInterrupt:
            print("\n(interrupted — cancelling pending, printing partial summary)\n",
                  flush=True)
            for f in futures:
                f.cancel()

    n_ok = sum(1 for r in results.values() if _race_status(r) == "OK")
    print(f"\n{n_ok}/{len(results)} fully passed")
    return 0


def _scope_paths_rel() -> list[str]:
    return [str(p.relative_to(runner.REPO)) for p in runner.SCOPE_FILES]


def cmd_revert(_: argparse.Namespace) -> int:
    paths = _scope_paths_rel()
    subprocess.run(["git", "checkout", "--", *paths], cwd=str(runner.REPO), check=True)
    print(f"reverted: {', '.join(paths)}")
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    paths = _scope_paths_rel()
    subprocess.run(["git", "add", *paths], cwd=str(runner.REPO), check=True)
    subprocess.run(
        ["git", "commit", "-m", args.message], cwd=str(runner.REPO), check=True
    )
    print(f"committed: {args.message}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="auto_race.race")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_show = sub.add_parser("show", help="print seeds, baseline, recent rows")
    sp_show.set_defaults(func=cmd_show)

    sp_run = sub.add_parser("run", help="run the full eval suite, log a row")
    sp_run.add_argument("--note", default="")
    sp_run.add_argument("--timeout", type=float, default=300.0,
                        help="per-seed timeout in seconds (default 300)")
    sp_run.set_defaults(func=cmd_run)

    sp_seed = sub.add_parser("seed", help="run one race for the named seed (no log)")
    sp_seed.add_argument("--name", required=True)
    sp_seed.add_argument("--timeout", type=float, default=300.0)
    sp_seed.set_defaults(func=cmd_seed)

    sp_all = sub.add_parser(
        "run-all",
        help="run every environment in main.py and print a per-env summary",
    )
    sp_all.add_argument(
        "--names", nargs="*", default=None,
        help="optional subset of env names; defaults to every key in main.py's "
             "environments dict",
    )
    sp_all.add_argument(
        "--timeout", type=float, default=60.0,
        help="per-env wall-clock cap in seconds (default 30); runs that "
             "exceed it are marked TIMEOUT and we move on",
    )
    sp_all.add_argument(
        "-j", "--parallel", type=int, default=1,
        help="number of races to run concurrently (default 1)",
    )
    sp_all.set_defaults(func=cmd_run_all)

    sp_revert = sub.add_parser("revert", help="git checkout in-scope source files")
    sp_revert.set_defaults(func=cmd_revert)

    sp_commit = sub.add_parser("commit", help="commit in-scope source files")
    sp_commit.add_argument("-m", "--message", required=True)
    sp_commit.set_defaults(func=cmd_commit)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
