"""Headless experiment runner for PID autoresearch.

Patches gains/limits/tuning_level inside
``controllers/main/exercises/ex1_pid_control.py``, runs the headless tune or
race shell scripts, and parses their stdout into a metrics dict. Designed to
be driven by an LLM agent or a small loop script — it does not own the
keep/revert decision, only the mechanics of one experiment.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PID_FILE = REPO / "controllers" / "main" / "exercises" / "ex1_pid_control.py"
TUNE_SH = REPO / "run_tune.sh"
RACE_SH = REPO / "run_race.sh"

LOOPS = ("rate_rp", "rate_y", "att_rp", "att_y",
         "vel_xy", "vel_z", "pos_xy", "pos_z")

GAIN_KEYS = tuple(f"{k}_{loop}" for loop in LOOPS for k in ("P", "I", "D"))

LIMIT_KEYS = ("L_rate_rp", "L_rate_y", "L_acc_rp", "L_vel_z", "L_vel_xy")


# ---------------- File patching ---------------------------------------------

def _read_pid_file() -> str:
    return PID_FILE.read_text()


def _write_pid_file(text: str) -> None:
    PID_FILE.write_text(text)


def current_gains() -> dict[str, float]:
    text = _read_pid_file()
    out: dict[str, float] = {}
    for key in GAIN_KEYS:
        m = re.search(rf'"{re.escape(key)}":\s*([-+0-9.eE]+)', text)
        if not m:
            raise ValueError(f"could not find gain {key} in {PID_FILE}")
        out[key] = float(m.group(1))
    return out


def current_limits() -> dict[str, float]:
    text = _read_pid_file()
    out: dict[str, float] = {}
    for key in LIMIT_KEYS:
        # Limits include numpy expressions like np.pi/6, so accept any expression
        # up to the trailing comma or closing brace.
        m = re.search(rf'"{re.escape(key)}":\s*([^,\n}}]+)', text)
        if not m:
            raise ValueError(f"could not find limit {key} in {PID_FILE}")
        out[key] = m.group(1).strip()  # type: ignore[assignment]
    return out  # type: ignore[return-value]


def current_tuning_level() -> str:
    text = _read_pid_file()
    # The non-racing default is the second occurrence of self.tuning_level = "..."
    matches = re.findall(r'self\.tuning_level\s*=\s*"([^"]+)"', text)
    if len(matches) < 2:
        raise ValueError("could not find non-racing tuning_level assignment")
    return matches[1]


def set_gains(updates: dict[str, float]) -> None:
    text = _read_pid_file()
    for key, val in updates.items():
        if key not in GAIN_KEYS:
            raise ValueError(f"unknown gain key: {key}")
        pattern = rf'("{re.escape(key)}":\s*)([-+0-9.eE]+)'
        new_text, n = re.subn(pattern, lambda m: f"{m.group(1)}{val}", text, count=1)
        if n == 0:
            raise ValueError(f"failed to patch gain {key}")
        text = new_text
    _write_pid_file(text)


def set_tuning_level(level: str) -> None:
    if level not in (*LOOPS, "off"):
        raise ValueError(f"unknown tuning_level: {level}")
    text = _read_pid_file()
    # Replace only the second `self.tuning_level = "..."` (the non-racing one).
    occurrences = list(re.finditer(r'(self\.tuning_level\s*=\s*")([^"]+)(")', text))
    if len(occurrences) < 2:
        raise ValueError("could not find non-racing tuning_level assignment")
    target = occurrences[1]
    text = text[:target.start()] + f'{target.group(1)}{level}{target.group(3)}' + text[target.end():]
    _write_pid_file(text)


# ---------------- Subprocess execution --------------------------------------

@dataclass
class TuneResult:
    tuning_level: str = ""
    ylabel: str = ""
    step_high: float = math.nan
    step_low: float = math.nan
    half_amp: float = math.nan
    ss_err_high_pct: float = math.nan
    ss_err_low_pct: float = math.nan
    overshoot_high_pct: float = math.nan
    overshoot_low_pct: float = math.nan
    rise_time_high_s: float = math.nan
    rise_time_low_s: float = math.nan
    samples: int = 0
    raw_stdout: str = ""

    @property
    def cost(self) -> float:
        rt = [self.rise_time_high_s, self.rise_time_low_s]
        if any(math.isnan(x) for x in rt):
            return math.inf
        os_ = [abs(self.overshoot_high_pct), abs(self.overshoot_low_pct)]
        ss = [abs(self.ss_err_high_pct), abs(self.ss_err_low_pct)]
        return float(sum(rt) / 2 + 0.02 * sum(os_) / 2 + 0.05 * sum(ss) / 2)


@dataclass
class RaceResult:
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
    def cost(self) -> float:
        if self.crashed or self.timed_out:
            return math.inf
        if len(self.lap_times) < 3:
            return math.inf
        if any(t >= 999.0 for t in self.lap_times):
            return math.inf
        if not self.all_gates_passed:
            return math.inf
        return float(self.lap_times[1] + self.lap_times[2])


_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|nan"


def _parse_tune_stdout(stdout: str) -> TuneResult:
    r = TuneResult(raw_stdout=stdout)
    if "===== PID TUNING RESULTS =====" not in stdout:
        return r
    block = stdout.split("===== PID TUNING RESULTS =====", 1)[1]

    def grab(pat: str) -> str | None:
        m = re.search(pat, block)
        return m.group(1) if m else None

    def gf(pat: str) -> float:
        v = grab(pat)
        try:
            return float(v) if v is not None else math.nan
        except ValueError:
            return math.nan

    r.tuning_level = grab(r"tuning_level:\s*(\S+)") or ""
    r.ylabel = grab(r"ylabel:\s*(.+)") or ""
    r.step_high = gf(rf"step_high:\s*({_FLOAT})")
    r.step_low = gf(rf"step_low:\s*({_FLOAT})")
    r.half_amp = gf(rf"half_amp:\s*({_FLOAT})")
    r.ss_err_high_pct = gf(rf"steady_state_error_high_pct:\s*({_FLOAT})")
    r.ss_err_low_pct = gf(rf"steady_state_error_low_pct:\s*({_FLOAT})")
    r.overshoot_high_pct = gf(rf"overshoot_high_pct:\s*({_FLOAT})")
    r.overshoot_low_pct = gf(rf"overshoot_low_pct:\s*({_FLOAT})")
    r.rise_time_high_s = gf(rf"rise_time_high_s:\s*({_FLOAT})")
    r.rise_time_low_s = gf(rf"rise_time_low_s:\s*({_FLOAT})")
    samples = grab(r"samples:\s*(\d+)")
    r.samples = int(samples) if samples else 0
    return r


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
    # Gate progress is printed as e.g.
    #   Gate progress: [[True, True, ...], [True, ...], [True, ...]]
    gp_match = re.search(r"Gate progress:\s*(\[\[.*?\]\])", stdout, re.DOTALL)
    if gp_match:
        # Convert Python-literal True/False text to a nested bool list without eval.
        raw = gp_match.group(1)
        try:
            import ast
            r.gate_progress = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            r.gate_progress = []
    return r


def run_tune(timeout: float = 120.0) -> TuneResult:
    proc = subprocess.run(
        [str(TUNE_SH)],
        cwd=str(REPO),
        capture_output=True, text=True,
        timeout=timeout,
    )
    return _parse_tune_stdout(proc.stdout + "\n" + proc.stderr)


def run_race(timeout: float = 300.0) -> RaceResult:
    try:
        proc = subprocess.run(
            [str(RACE_SH)],
            cwd=str(REPO),
            capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        r = _parse_race_stdout(out)
        r.timed_out = True
        return r
    return _parse_race_stdout(proc.stdout + "\n" + proc.stderr)


# ---------------- Logging ---------------------------------------------------

RESULTS_TSV = REPO / "auto_tune" / "results.tsv"

RESULTS_HEADER = (
    "ts\tkind\tcommit\tloop\tgains_changed\tcost\trise_high\trise_low\t"
    "overshoot_high\tovershoot_low\tss_high\tss_low\tlap_times\tcrashed\tgates_ok\tnote\n"
)


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), text=True
        )
        return out.strip()
    except subprocess.CalledProcessError:
        return ""


def append_tune_row(loop: str, gains_changed: dict[str, float], result: TuneResult, note: str = "") -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER)
    import datetime
    row = "\t".join([
        datetime.datetime.now().isoformat(timespec="seconds"),
        "tune",
        _git_head(),
        loop,
        ",".join(f"{k}={v}" for k, v in gains_changed.items()),
        f"{result.cost:.4f}",
        f"{result.rise_time_high_s:.3f}",
        f"{result.rise_time_low_s:.3f}",
        f"{result.overshoot_high_pct:.2f}",
        f"{result.overshoot_low_pct:.2f}",
        f"{result.ss_err_high_pct:.2f}",
        f"{result.ss_err_low_pct:.2f}",
        "",
        "",
        "",
        note.replace("\t", " "),
    ]) + "\n"
    with RESULTS_TSV.open("a") as f:
        f.write(row)


def append_race_row(gains_changed: dict[str, float], result: RaceResult, note: str = "") -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER)
    import datetime
    row = "\t".join([
        datetime.datetime.now().isoformat(timespec="seconds"),
        "race",
        _git_head(),
        "",
        ",".join(f"{k}={v}" for k, v in gains_changed.items()),
        f"{result.cost:.4f}",
        "", "", "", "", "", "",
        ",".join(f"{t:.2f}" for t in result.lap_times),
        "1" if result.crashed else "0",
        "1" if result.all_gates_passed else "0",
        note.replace("\t", " "),
    ]) + "\n"
    with RESULTS_TSV.open("a") as f:
        f.write(row)


if __name__ == "__main__":
    print("current tuning_level:", current_tuning_level(), file=sys.stderr)
    print("current gains:", file=sys.stderr)
    for k, v in current_gains().items():
        print(f"  {k} = {v}", file=sys.stderr)
