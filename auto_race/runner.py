"""Headless racing experiment runner for end-to-end race autoresearch.

Patches main.py's ``CLI_MODE`` and ``env_name``, runs ``./run_race.sh`` once
per seed in :data:`SEEDS`, parses stdout into per-race and suite metrics, and
appends rows to ``auto_race/results.tsv``. main.py is restored after every
run via a context manager so the working tree is unaffected.

Designed to be driven by an LLM agent or a small loop script. The keep/revert
decision lives outside the runner — see ``program.md``.
"""

from __future__ import annotations

import datetime
import math
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAIN_FILE = REPO / "controllers" / "main" / "main.py"
RACE_SH = REPO / "run_race.sh"
RESULTS_TSV = REPO / "auto_race" / "results.tsv"

SCOPE_FILES = (
    REPO / "controllers" / "main" / "assignment" / "my_assignment.py",
    REPO / "controllers" / "main" / "exercises" / "ex1_pid_control.py",
)

SEEDS = (
    "race_skip_gate_0",
    "orthogonal_gate_1",
    "skip_gate_3",
)


# ---------------- main.py patching -----------------------------------------

@contextmanager
def patched_main(env_name: str):
    """Set ``CLI_MODE = True`` and ``env_name = "<env_name>"`` in main.py for
    the duration of the with-block. Restores the original file contents on
    exit, regardless of how the block exits.
    """
    original = MAIN_FILE.read_text()
    text = re.sub(
        r'^CLI_MODE\s*=\s*\w+',
        'CLI_MODE = True',
        original,
        count=1,
        flags=re.MULTILINE,
    )
    if 'CLI_MODE = True' not in text:
        raise ValueError("failed to patch CLI_MODE in main.py")
    text = re.sub(
        r'^env_name\s*=\s*"[^"]*"',
        f'env_name = "{env_name}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if f'env_name = "{env_name}"' not in text:
        raise ValueError(f"failed to patch env_name to {env_name} in main.py")
    MAIN_FILE.write_text(text)
    try:
        yield
    finally:
        MAIN_FILE.write_text(original)


# ---------------- Result types ---------------------------------------------

@dataclass
class RaceResult:
    seed: str = ""
    lap_times: list[float] = field(default_factory=list)
    gate_progress: list[list[bool]] = field(default_factory=list)
    crashed: bool = False
    timed_out: bool = False
    raw_stdout: str = ""

    @property
    def all_gates_passed(self) -> bool:
        if not self.gate_progress:
            return False
        return all(all(lap) for lap in self.gate_progress)

    @property
    def lap1_ok(self) -> bool:
        return len(self.lap_times) >= 1 and self.lap_times[0] < 240.0

    @property
    def cost(self) -> float:
        if self.crashed or self.timed_out:
            return math.inf
        if len(self.lap_times) < 3:
            return math.inf
        if any(t >= 999.0 for t in self.lap_times):
            return math.inf
        if not self.all_gates_passed:
            return math.inf
        if not self.lap1_ok:
            return math.inf
        return float(self.lap_times[1] + self.lap_times[2])


@dataclass
class SuiteResult:
    races: list[RaceResult] = field(default_factory=list)

    @property
    def all_finite(self) -> bool:
        return bool(self.races) and all(math.isfinite(r.cost) for r in self.races)

    @property
    def cost(self) -> float:
        if not self.all_finite:
            return math.inf
        return float(sum(r.cost for r in self.races) / len(self.races))

    @property
    def worst_cost(self) -> float:
        if not self.races:
            return math.inf
        return float(max(r.cost for r in self.races))


# ---------------- Parsing --------------------------------------------------

def _parse_race_stdout(stdout: str) -> RaceResult:
    r = RaceResult(raw_stdout=stdout)
    r.crashed = "Crash detected" in stdout
    matches = re.findall(r"Lap times:\s*\[([^\]]+)\]", stdout)
    if matches:
        nums = [s.strip() for s in matches[-1].split(",")]
        try:
            r.lap_times = [float(x) for x in nums]
        except ValueError:
            r.lap_times = []
    gp_match = re.search(r"Gate progress:\s*(\[\[.*?\]\])", stdout, re.DOTALL)
    if gp_match:
        try:
            import ast
            r.gate_progress = ast.literal_eval(gp_match.group(1))
        except (ValueError, SyntaxError):
            r.gate_progress = []
    return r


# ---------------- Subprocess execution -------------------------------------

def run_race_for_seed(seed: str, timeout: float = 300.0) -> RaceResult:
    """Patch main.py for ``seed``, run one race, parse stdout. Restores
    main.py on every exit path."""
    with patched_main(seed):
        try:
            proc = subprocess.run(
                [str(RACE_SH)],
                cwd=str(REPO),
                capture_output=True, text=True,
                timeout=timeout,
            )
            r = _parse_race_stdout(proc.stdout + "\n" + proc.stderr)
        except subprocess.TimeoutExpired as e:
            out = e.stdout or ""
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            r = _parse_race_stdout(out)
            r.timed_out = True
    r.seed = seed
    return r


def run_suite(seeds: tuple[str, ...] = SEEDS, timeout_per_seed: float = 300.0) -> SuiteResult:
    """Run every seed sequentially. Aborts early on the first failure
    (crash / missed gate / timeout) since the suite cost is already inf and
    later seeds add no information for the keep/revert decision."""
    races: list[RaceResult] = []
    for seed in seeds:
        print(f"[suite] running seed: {seed}", file=sys.stderr, flush=True)
        r = run_race_for_seed(seed, timeout=timeout_per_seed)
        cost_str = "inf" if not math.isfinite(r.cost) else f"{r.cost:.2f}"
        print(
            f"[suite] seed={seed} cost={cost_str} laps={r.lap_times} "
            f"crashed={r.crashed} timed_out={r.timed_out} "
            f"gates_ok={r.all_gates_passed}",
            file=sys.stderr, flush=True,
        )
        races.append(r)
        if not math.isfinite(r.cost):
            print(
                f"[suite] aborting after seed '{seed}' (cost=inf)",
                file=sys.stderr, flush=True,
            )
            break
    return SuiteResult(races=races)


# ---------------- Logging --------------------------------------------------

RESULTS_HEADER = "ts\tcommit\tsuite_cost\tworst_cost\tper_seed\tnote\n"


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), text=True
        )
        return out.strip()
    except subprocess.CalledProcessError:
        return ""


def _format_per_seed(suite: SuiteResult) -> str:
    parts = []
    for r in suite.races:
        cost_str = "inf" if not math.isfinite(r.cost) else f"{r.cost:.2f}"
        laps = ",".join(f"{t:.1f}" for t in r.lap_times) or "-"
        parts.append(
            f"{r.seed}={cost_str}(laps={laps};crash={int(r.crashed)};"
            f"to={int(r.timed_out)};gates={int(r.all_gates_passed)})"
        )
    return ";".join(parts)


def append_suite_row(suite: SuiteResult, note: str = "") -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER)
    cost_str = "inf" if not math.isfinite(suite.cost) else f"{suite.cost:.4f}"
    worst_str = "inf" if not math.isfinite(suite.worst_cost) else f"{suite.worst_cost:.4f}"
    row = "\t".join([
        datetime.datetime.now().isoformat(timespec="seconds"),
        _git_head(),
        cost_str,
        worst_str,
        _format_per_seed(suite),
        note.replace("\t", " ").replace("\n", " "),
    ]) + "\n"
    with RESULTS_TSV.open("a") as f:
        f.write(row)


# ---------------- Baseline / per-seed lookup -------------------------------

def best_known_suite_cost() -> float:
    """Lowest finite ``suite_cost`` seen so far in ``results.tsv``."""
    if not RESULTS_TSV.exists():
        return math.inf
    best = math.inf
    for line in RESULTS_TSV.read_text().splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        try:
            cost = float(cols[2])
        except ValueError:
            continue
        if math.isfinite(cost) and cost < best:
            best = cost
    return best


def best_known_per_seed() -> dict[str, float]:
    """Lowest finite per-seed cost seen so far, parsed from the per_seed
    column. Useful for the no-regression rule in program.md."""
    out: dict[str, float] = {s: math.inf for s in SEEDS}
    if not RESULTS_TSV.exists():
        return out
    pat = re.compile(r"([A-Za-z0-9_]+)=([0-9.]+|inf)\(")
    for line in RESULTS_TSV.read_text().splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        for seed, cost_s in pat.findall(cols[4]):
            if cost_s == "inf":
                continue
            try:
                cost = float(cost_s)
            except ValueError:
                continue
            if seed in out and cost < out[seed]:
                out[seed] = cost
    return out


if __name__ == "__main__":
    print(f"seeds: {', '.join(SEEDS)}", file=sys.stderr)
    print(f"best known suite_cost: {best_known_suite_cost()}", file=sys.stderr)
    print(f"best per-seed: {best_known_per_seed()}", file=sys.stderr)
