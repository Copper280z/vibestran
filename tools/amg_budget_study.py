#!/usr/bin/env python3
"""Constant against rigid AMG at identical absolute device budgets.

Priority 1 of PCG_AMG_RESEARCH.md. Rigid needs about twice the device memory, so a
comparison of resident times flatters it. Each class runs at budgets that span the
conditions a user meets as a deck outgrows the card:

  both-resident   both fit
  rigid-90/75/50  constant fits while that share of rigid's fine rows stays on the GPU
  rigid-host      rigid's fixed state does not fit, so its hierarchy stays on the host
  both-host       neither fits

`screen` runs each cell once to find the comparisons worth repeating. `paired` repeats
them with alternating order, keeping every sample, and can add `pcg_benchmark` runs
that time setup, retained-factor solves and refactoring separately. `large` runs decks
sized against the card's own usable memory with no forced budget, to show what forced
budgets cannot: transient and driver allocations at real sizes. `report` gives
medians, ranges and the verdicts below, and exits nonzero when a run's execution mode
was not the one its budget should produce.

Criteria, fixed before these measurements. A default must solve every class at every
budget where either method solves, and at equal budgets stay within 2x of the better
method's complete-deck time and 4x of its solver time. Budget conditions where neither
method solves within the time limit are reported and do not count against either.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import amg_model_suite as suite

MODES = ("constant", "rigid")
CONDITIONS = ("both-resident", "rigid-90", "rigid-75", "rigid-50", "rigid-host", "both-host")
# The finest level of each class the research plan names, each about a million equations.
DEFAULT_CLASSES = {"solid-hex": 40, "solid-tet10": 19, "shell-thin": 400, "shell-membrane": 400,
                   "space-frame": 52, "rbe-assembly": 192}
MIB = 1024 * 1024
MAX_COMPLETE_SLOWDOWN = 2.0
MAX_SOLVER_SLOWDOWN = 4.0
# F06 values carry seven significant digits, so this is as close as the tables can show.
SAME_SOLUTION = 1e-5


def budget_conditions(constant, rigid):
    """The budget in MiB of each condition, from each mode's plan with memory to spare."""
    up = lambda bytes_: math.ceil(bytes_ / MIB)
    preferred = lambda plan: plan.get("fp32_planned_bytes", plan["planned_bytes"])
    return {
        "both-resident": up(1.02 * max(preferred(constant), preferred(rigid))),
        "rigid-90": up(rigid["row_budgets"]["90"]),
        "rigid-75": up(rigid["row_budgets"]["75"]),
        "rigid-50": up(rigid["row_budgets"]["50"]),
        "rigid-host": math.floor(0.95 * rigid["fixed_bytes"] / MIB),
        "both-host": math.floor(0.9 * min(constant["fixed_bytes"], rigid["fixed_bytes"]) / MIB),
    }


def expected_execution(plan, budget_mib):
    """What the planner must choose for a hierarchy with this plan at this budget."""
    budget = budget_mib * MIB
    if budget >= plan["planned_bytes"]:
        return "resident"
    return "spill" if budget >= plan["fixed_bytes"] else "host"


def execution_key(plan, budget_mib):
    """Runs with the same key execute alike, unless they spill. A mixed solve resident in fp64 is not one resident in fp32."""
    return expected_execution(plan, budget_mib), budget_mib * MIB >= plan.get("fp32_planned_bytes", 0)


def own_plan_problem(record):
    """Why a run with no forced budget broke its logged plan, or None.

    A spilled run logs the partition it chose rather than what staying resident would
    need, so its log cannot say whether it had to spill, only that its choice fits.
    """
    if record["outcome"] != "solved" or record.get("execution") == "host":
        return None
    if "planned_bytes" not in record or "budget_bytes" not in record:
        return "solved with no plan in the log"
    if record["planned_bytes"] > record["budget_bytes"]:
        return f"planned {record['planned_bytes']} bytes over a budget of {record['budget_bytes']}"
    return None


def execution_problem(record, expected):
    """Why a solved run did not execute as planned, or None. A failed solve has its own outcome."""
    if record["outcome"] != "solved" or record.get("execution") == expected:
        return None
    return f"executed as {record.get('execution')}, expected {expected}"


def summarize(values):
    """Median and range. Every sample is kept in the results; this only reports them."""
    return {"median": statistics.median(values), "low": min(values), "high": max(values), "samples": len(values)}


def paired_ratios(samples):
    """rigid / constant for each repeat where both solved, from {repeat: {mode: seconds}}."""
    return [pair["rigid"] / pair["constant"] for pair in samples.values()
            if "rigid" in pair and "constant" in pair and pair["constant"] > 0]


def verdict(ratios):
    """Which mode won a cell. A range of ratios that spans 1 is not a win for either."""
    if not ratios:
        return "unpaired"
    if min(ratios) > 1:
        return "constant"
    return "rigid" if max(ratios) < 1 else "tie"


def criteria_failures(cells):
    """Which mode breaks the criteria, from {(class, condition): {mode: {outcome, complete_s, solver_s}}}."""
    failures = {mode: [] for mode in MODES}
    for (name, condition), results in sorted(cells.items()):
        solved = {mode: r for mode, r in results.items() if r["outcome"] == "solved"}
        if not solved:
            continue
        best_complete = min(r["complete_s"] for r in solved.values())
        best_solver = min(r["solver_s"] for r in solved.values())
        for mode in MODES:
            result = solved.get(mode)
            if result is None:
                failures[mode].append(f"{name} {condition}: {results.get(mode, {}).get('outcome', 'not run')}")
                continue
            if result["complete_s"] > MAX_COMPLETE_SLOWDOWN * best_complete:
                failures[mode].append(f"{name} {condition}: complete {result['complete_s'] / best_complete:.2f}x")
            if result["solver_s"] > MAX_SOLVER_SLOWDOWN * best_solver:
                failures[mode].append(f"{name} {condition}: solver {result['solver_s'] / best_solver:.2f}x")
    return failures


class VramSampler:
    """Peak device memory above what was in use before the run, from nvidia-smi."""
    QUERY = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]

    def __enter__(self):
        self.peak = None
        self.samples = tempfile.TemporaryFile("w+")
        try:
            self.before = int(subprocess.run(self.QUERY, capture_output=True, text=True, timeout=30).stdout.split()[0])
            self.process = subprocess.Popen([*self.QUERY, "-lms", "200"], stdout=self.samples, stderr=subprocess.DEVNULL)
        except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
            self.process = None
        return self

    def __exit__(self, *_):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.samples.seek(0)
            used = [int(line) for line in self.samples.read().split() if line.isdigit()]
            self.peak = max(used) - self.before if used else None
        self.samples.close()


def run_phases(binary, deck, backend, mode, budget_mib, args):
    """One pcg_benchmark run: factor, solves and refactor timed apart."""
    command = [*args.prefix.split(), str(binary), "--structural-deck", str(deck), "--amg-modes", mode,
               "--backend", backend, "--budget-mib", str(budget_mib), "--fp64"]
    env = dict(os.environ)
    env.pop("VIBESTRAN_PCG_DEVICE_MEMORY_MIB", None)
    env.pop("VIBESTRAN_AMG_PROBE", None)
    with tempfile.TemporaryFile("w+") as out:
        outcome, detail, status, wall, peak = suite.run_and_measure(command, env, args.timeout, stdout=out)
        out.seek(0)
        lines = out.read().strip().splitlines()
    record = {"outcome": outcome, "status": detail or outcome, "exit": status, "wall_s": wall, "peak_rss_bytes": peak,
              "command": " ".join(command), "budget_mib": budget_mib}
    if outcome == "solved" and lines:
        record.update(json.loads(lines[-1]))
    return record


class Study:
    def __init__(self, args):
        self.args = args
        self.args.no_probe = True
        self.args.budget_mib = 0
        self.results = open(args.results, "a")
        self.problems = []

    def deck(self, name):
        directory = self.args.work / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "deck.bdf"
        if not path.exists():
            suite.write_deck(name, CLASSES_IN_USE[name], path)
        return path

    def run(self, name, condition, budget_mib, mode, stage, repeat, order, plan, reference):
        args = self.args
        args.budget_mib = budget_mib
        deck = self.deck(name)
        label = f"-{condition}-{stage}{repeat}"
        with VramSampler() as vram:
            record, _, tables = suite.run_once(args.binary, deck, args.backend, mode, args, label=label)
        expected = None
        if plan is not None:
            expected = expected_execution(plan, budget_mib) if budget_mib else "resident"
        row = {"stage": stage, "class": name, "refinement": CLASSES_IN_USE[name], "condition": condition,
               "modes": mode, "backend": args.backend, "repeat": repeat, "order": order, "expected_execution": expected,
               "peak_vram_mib": vram.peak, **record}
        problem = own_plan_problem(record) if plan is None else execution_problem(record, expected)
        if problem:
            row["execution_problem"] = problem
            self.problems.append(f"{name} {condition} {mode}: {problem}")
        displacements = tables.get("displacements")
        if record["outcome"] == "solved" and displacements:
            if reference is not None and name not in reference:
                reference[name] = displacements
            if reference is not None:
                row["difference_from_reference"] = suite.relative_difference(displacements, reference[name])
                if row["difference_from_reference"] > SAME_SOLUTION:
                    self.problems.append(f"{name} {condition} {mode}: solution differs by {row['difference_from_reference']:.2e}")
        self.results.write(json.dumps(row) + "\n")
        self.results.flush()
        print(f"{stage} {name} {condition} {budget_mib} MiB {mode} #{repeat}: {record['outcome']} "
              f"{row.get('execution')} rows={row.get('gpu_rows')}/{row.get('cpu_rows')} it={row.get('iterations')} "
              f"setup={row.get('setup_s')} iterate={row.get('iterate_s')} wall={row['wall_s']:.1f} "
              f"vram={row['peak_vram_mib']} rss={row['peak_rss_bytes'] / MIB:.0f}", flush=True)
        return row

    def plans(self, name, reference):
        """Each mode's plan with memory to spare; these runs also count as both-resident samples."""
        plans = {}
        for mode in MODES:
            row = self.run(name, "unconstrained", 0, mode, "plan", 0, 0, {}, reference)
            if "planned_bytes" not in row:
                self.problems.append(f"{name} {mode}: no plan in the log ({row['outcome']}); class skipped")
                return None
            plans[mode] = {key: row.get(key) for key in ("planned_bytes", "fixed_bytes", "row_budgets")}
            if row.get("fp32_hierarchy"):
                plans[mode] = self.fp64_plan(name, mode, row, reference)
                if plans[mode] is None:
                    return None
        return plans

    def fp64_plan(self, name, mode, fp32, reference):
        """A mixed solve keeps its fp32 hierarchy only while it fits, then plans in fp64.

        Fp32 fine rows do not spill, so only a run just under the fp32 plan logs the fp64
        plan the budgets below it depend on. That plan can be the smaller of the two: the
        fp32 hierarchy adds to the fp64 fine operator refinement needs.
        """
        budget = math.floor(fp32["planned_bytes"] / MIB) - 1
        row = self.run(name, "under-fp32", budget, mode, "plan", 1, 0, None, reference)
        if "row_budgets" not in row:
            self.problems.append(f"{name} {mode}: no fp64 plan under the fp32 one ({row['outcome']}); class skipped")
            return None
        fits = row.get("execution") == "resident"
        return {"fp32_planned_bytes": fp32["planned_bytes"],
                "planned_bytes": min(fp32["planned_bytes"], row["planned_bytes"]) if fits else fp32["planned_bytes"],
                "fixed_bytes": row["fixed_bytes"], "row_budgets": row["row_budgets"]}

    def close(self):
        self.results.close()
        if self.problems:
            print("\nCHECKS FAILED:", *self.problems, sep="\n  ", file=sys.stderr)
            sys.exit(1)


CLASSES_IN_USE = dict(DEFAULT_CLASSES)


def chosen_classes(args):
    for specification in args.deck:
        suite.add_external_deck(specification)
        CLASSES_IN_USE[specification.split("=")[0]] = 1
    return args.classes.split(",") if args.classes else list(CLASSES_IN_USE)


def cmd_screen(args):
    """Each cell once. A mode whose execution is the same as at an earlier budget is not rerun."""
    study = Study(args)
    budgets = {}
    if args.manifest:
        suite.write_manifest(args)
    for name in chosen_classes(args):
        reference = {}
        plans = study.plans(name, reference)
        if plans is None:
            (args.work / name / "deck.bdf").unlink(missing_ok=True)
            continue
        conditions = budget_conditions(plans["constant"], plans["rigid"])
        budgets[name] = {"plans": plans, "budgets": conditions}
        args.budgets.write_text(json.dumps(budgets, indent=2) + "\n")
        for mode in MODES:
            ran = {("resident", True): "unconstrained"}
            for condition in CONDITIONS:
                budget = conditions[condition]
                expected = expected_execution(plans[mode], budget)
                key = execution_key(plans[mode], budget)
                # Spilled rows change with the budget; resident and host states do not.
                if key in ran and expected != "spill":
                    row = {"stage": "screen", "class": name, "condition": condition, "modes": mode, "budget_mib": budget,
                           "expected_execution": expected, "same_as": ran[key]}
                    study.results.write(json.dumps(row) + "\n")
                    study.results.flush()
                    continue
                study.run(name, condition, budget, mode, "screen", 0, 0, plans[mode], reference)
                ran[key] = condition
        (args.work / name / "deck.bdf").unlink(missing_ok=True)
    study.close()


def cmd_paired(args):
    """Paired repeats of chosen cells, alternating which mode runs first."""
    study = Study(args)
    known = json.loads(args.budgets.read_text())
    for specification in args.deck:
        suite.add_external_deck(specification)
        CLASSES_IN_USE[specification.split("=")[0]] = 1
    cells = [cell.split(":") for cell in args.cells.split(",")]
    for name in dict.fromkeys(cell[0] for cell in cells):
        reference = {}
        for _, condition in [cell for cell in cells if cell[0] == name]:
            budget = known[name]["budgets"][condition]
            plans = known[name]["plans"]
            for repeat in range(args.repeats):
                for order, mode in enumerate(MODES if repeat % 2 == 0 else reversed(MODES)):
                    study.run(name, condition, budget, mode, "paired", repeat, order, plans[mode], reference)
                    if args.pcg_benchmark:
                        phases = run_phases(args.pcg_benchmark, study.deck(name), args.backend, mode, budget, args)
                        row = {"stage": "phases", "class": name, "condition": condition, "repeat": repeat, "order": order,
                               "expected_execution": expected_execution(plans[mode], budget), **phases, "modes": mode}
                        problem = execution_problem(phases, row["expected_execution"])
                        if problem:
                            row["execution_problem"] = problem
                            study.problems.append(f"{name} {condition} {mode} phases: {problem}")
                        study.results.write(json.dumps(row) + "\n")
                        study.results.flush()
                        print(f"phases {name} {condition} {mode} #{repeat}: {phases['outcome']} "
                              + " ".join(f"{k}={phases.get(k)}" for k in ("execution", "factor_s", "first_solve_s",
                                                                           "retained_solve_s", "refactor_s", "refactored_solve_s")),
                              flush=True)
        (args.work / name / "deck.bdf").unlink(missing_ok=True)
    study.close()


def cmd_large(args):
    """Decks at chosen sizes with the card's own usable memory as the budget, alternating order."""
    study = Study(args)
    for cell in args.cells.split(","):
        name, refinement = cell.split(":")
        CLASSES_IN_USE[name] = int(refinement)
        reference = {}
        for repeat in range(args.repeats):
            for order, mode in enumerate(MODES if repeat % 2 == 0 else reversed(MODES)):
                study.run(name, f"large-{refinement}", 0, mode, "large", repeat, order, None, reference)
        (args.work / name / "deck.bdf").unlink(missing_ok=True)
    study.close()


def report_large(rows):
    """Plan, measured and peak memory of each large run. VRAM above the budget is what the plan does not count."""
    print("\nLarge decks, no forced budget:")
    print("| Class | Refinement | Equations | Mode | # | Outcome | Execution | GPU rows | Budget MiB | Chosen plan MiB | Measured MiB "
          "| Peak VRAM | Peak RSS | Complete s | Solver s | Iterations |")
    print("|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    mib = lambda row, key: f"{row[key] / MIB:.0f}" if row.get(key) is not None else "-"
    for row in rows:
        fraction = row["gpu_rows"] / row["n"] if row.get("gpu_rows") is not None and row.get("n") else None
        print(f"| {row['class']} | {row['refinement']} | {row.get('n') or '-'} | {row['modes']} | {row['repeat']} | {row['outcome']} | "
              f"{row.get('execution') or '-'} | {f'{fraction:.0%}' if fraction is not None else '-'} | {mib(row, 'budget_bytes')} | "
              f"{mib(row, 'planned_bytes')} | {mib(row, 'measured_bytes')} | {row.get('peak_vram_mib')} | {row['peak_rss_bytes'] / MIB:.0f} | "
              f"{row['wall_s']:.1f} | {solver_seconds(row):.1f} | {sum(row.get('iterations') or [])} |")


def solver_seconds(row):
    return row.get("setup_s", 0.0) + row.get("iterate_s", 0.0)


def cmd_report(args):
    rows = [json.loads(line) for path in args.results for line in path.read_text().splitlines()]
    # Large runs are rechecked, so a result file from an earlier version of the check reads the same.
    for row in rows:
        if row.get("stage") == "large":
            row["execution_problem"] = own_plan_problem(row)
    bad = [row for row in rows if row.get("execution_problem")]
    runs = [row for row in rows if row.get("stage") in ("screen", "paired") and "outcome" in row]
    # A cell the screen did not rerun has the result of the same execution at another budget.
    for row in rows:
        if "same_as" in row:
            runs += [{**source, "condition": row["condition"], "budget_mib": row["budget_mib"]} for source in rows
                     if source.get("stage") in ("screen", "plan") and "outcome" in source and source["class"] == row["class"]
                     and source["modes"] == row["modes"] and source["condition"] == row["same_as"]]
    by_cell = {}
    for row in runs:
        by_cell.setdefault((row["class"], CONDITIONS.index(row["condition"]), row["condition"]), {}).setdefault(row["modes"], []).append(row)
    print("| Class | Condition | MiB | Mode | Execution | GPU rows | n | Outcomes | Complete s | Solver s | Iterations | Peak VRAM | Peak RSS |")
    print("|---|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
    cells = {}
    for (name, _, condition), modes in sorted(by_cell.items()):
        for mode in MODES:
            samples = modes.get(mode, [])
            if not samples:
                continue
            solved = [row for row in samples if row["outcome"] == "solved"]
            outcomes = ",".join(sorted({row["outcome"] for row in samples}))
            if solved:
                complete, solver = summarize([r["wall_s"] for r in solved]), summarize([solver_seconds(r) for r in solved])
                last = solved[-1]
                fraction = last.get("gpu_rows", 0) / last["n"] if last.get("n") else 0
                print(f"| {name} | {condition} | {last['budget_mib']} | {mode} | {last.get('execution')}{'/fp32' if last.get('fp32_hierarchy') else ''} | {fraction:.0%} | {len(samples)} | "
                      f"{outcomes} | {complete['median']:.2f} ({complete['low']:.2f}-{complete['high']:.2f}) | "
                      f"{solver['median']:.2f} ({solver['low']:.2f}-{solver['high']:.2f}) | {sum(last['iterations'])} | "
                      f"{last.get('peak_vram_mib')} | {last['peak_rss_bytes'] / MIB:.0f} |")
                cells.setdefault((name, condition), {})[mode] = {"outcome": "solved", "complete_s": complete["median"], "solver_s": solver["median"]}
            else:
                print(f"| {name} | {condition} | {samples[-1]['budget_mib']} | {mode} | - | - | {len(samples)} | {outcomes} | - | - | - | - | - |")
                cells.setdefault((name, condition), {})[mode] = {"outcome": outcomes}
    print("\nPaired verdicts (rigid / constant, complete-deck time):")
    for (name, _, condition), modes in sorted(by_cell.items()):
        pairs = {}
        for mode in MODES:
            for row in modes.get(mode, []):
                if row["stage"] == "paired" and row["outcome"] == "solved":
                    pairs.setdefault(row["repeat"], {})[mode] = row["wall_s"]
        ratios = paired_ratios(pairs)
        if ratios:
            print(f"  {name} {condition}: {verdict(ratios)}; median {statistics.median(ratios):.2f}, "
                  f"range {min(ratios):.2f}-{max(ratios):.2f}, {len(ratios)} pairs")
    large = [row for row in rows if row.get("stage") == "large"]
    if large:
        report_large(large)
    phases = [row for row in rows if row.get("stage") == "phases" and row["outcome"] == "solved"]
    if phases:
        print("\nPhases, median seconds:")
        groups = {}
        for row in phases:
            groups.setdefault((row["class"], row["condition"], row["modes"]), []).append(row)
        for (name, condition, mode), group in sorted(groups.items()):
            parts = " ".join(f"{key}={statistics.median(r[key] for r in group):.2f}"
                             for key in ("factor_s", "first_solve_s", "retained_solve_s", "refactor_s", "refactored_solve_s"))
            print(f"  {name} {condition} {mode}: {group[-1]['execution']} {parts} n={len(group)}")
    if cells:
        print("\nCriteria:")
    for mode, failures in (criteria_failures(cells) if cells else {}).items():
        print(f"  {mode}: {'meets them' if not failures else ''}")
        for failure in failures:
            print(f"    {failure}")
    if bad:
        print("\nEXECUTION PROBLEMS:", *[f"{r['class']} {r['condition']} {r['modes']}: {r['execution_problem']}" for r in bad], sep="\n  ", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True, dest="command")
    for name, function in (("screen", cmd_screen), ("paired", cmd_paired), ("large", cmd_large)):
        run = commands.add_parser(name)
        run.set_defaults(function=function)
        run.add_argument("--binary", type=Path, required=True)
        run.add_argument("--work", type=Path, required=True)
        run.add_argument("--results", type=Path, required=True)
        if function is not cmd_large:
            run.add_argument("--budgets", type=Path, required=True, help="the budgets screen writes and paired reads")
        run.add_argument("--backend", default="cuda-pcg")
        run.add_argument("--classes", help="comma separated; the default is the research plan's")
        run.add_argument("--deck", action="append", default=[], help="NAME=PATH: also run this deck")
        run.add_argument("--archive", type=Path, help="keep each run's log here")
        run.add_argument("--prefix", default="")
        run.add_argument("--timeout", type=float, default=300)
    screen = [c for c in commands.choices.values() if c.get_default("function") is cmd_screen][0]
    screen.add_argument("--manifest", type=Path, help="write what reproduces this run")
    screen.add_argument("--source-commit")
    screen.add_argument("--source-patch", type=Path, help="the working-tree patch on top of the commit")
    screen.add_argument("--build-dir", type=Path, help="the Meson build directory of the binary, for the manifest")
    paired = [c for c in commands.choices.values() if c.get_default("function") is cmd_paired][0]
    paired.add_argument("--cells", required=True, help="comma-separated class:condition")
    paired.add_argument("--repeats", type=int, default=5)
    paired.add_argument("--pcg-benchmark", type=Path, help="also time each sample's phases with this binary")
    large = [c for c in commands.choices.values() if c.get_default("function") is cmd_large][0]
    large.add_argument("--cells", required=True, help="comma-separated class:refinement")
    large.add_argument("--repeats", type=int, default=2)
    report = commands.add_parser("report")
    report.set_defaults(function=cmd_report)
    report.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    if getattr(args, "work", None):
        args.work.mkdir(parents=True, exist_ok=True)
    args.function(args)


if __name__ == "__main__":
    main()
