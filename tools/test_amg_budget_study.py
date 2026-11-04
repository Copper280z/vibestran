#!/usr/bin/env python3
"""The study's budgets, execution checks and verdicts mean what they claim."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import amg_budget_study as study

MIB = study.MIB
CONSTANT = {"planned_bytes": 900 * MIB, "fixed_bytes": 500 * MIB, "row_budgets": {"50": 700 * MIB, "75": 800 * MIB, "90": 860 * MIB}}
RIGID = {"planned_bytes": 1900 * MIB, "fixed_bytes": 1200 * MIB, "row_budgets": {"50": 1500 * MIB, "75": 1700 * MIB, "90": 1820 * MIB}}


class Budgets(unittest.TestCase):
    def setUp(self):
        self.budgets = study.budget_conditions(CONSTANT, RIGID)

    def execution(self, plan, condition):
        return study.expected_execution(plan, self.budgets[condition])

    def test_each_condition_produces_the_state_it_names_for_rigid(self):
        self.assertEqual(self.execution(RIGID, "both-resident"), "resident")
        for condition in ("rigid-90", "rigid-75", "rigid-50"):
            self.assertEqual(self.execution(RIGID, condition), "spill", condition)
        self.assertEqual(self.execution(RIGID, "rigid-host"), "host")
        self.assertEqual(self.execution(RIGID, "both-host"), "host")

    def test_constant_stays_resident_where_rigid_spills_and_leaves_only_when_it_must(self):
        for condition in ("both-resident", "rigid-90", "rigid-50", "rigid-host"):
            self.assertEqual(self.execution(CONSTANT, condition), "resident", condition)
        self.assertEqual(self.execution(CONSTANT, "both-host"), "host")

    def test_budgets_shrink_in_condition_order(self):
        values = [self.budgets[c] for c in study.CONDITIONS]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(len(set(values)), len(values))

    def test_a_budget_rounded_to_mib_still_holds_its_rows(self):
        plan = dict(RIGID, row_budgets={"50": 1500 * MIB + 1, "75": 1700 * MIB + 1, "90": 1820 * MIB + 1})
        self.assertGreaterEqual(study.budget_conditions(CONSTANT, plan)["rigid-50"] * MIB, plan["row_budgets"]["50"])


class MixedPlans(unittest.TestCase):
    """The fp32 hierarchy adds to the fp64 fine operator, so it can need more than fp64 does."""
    PLAN = {**RIGID, "fp32_planned_bytes": 2000 * MIB}

    def test_both_resident_keeps_the_fp32_hierarchy(self):
        budget = study.budget_conditions(CONSTANT, self.PLAN)["both-resident"]
        self.assertGreaterEqual(budget * MIB, self.PLAN["fp32_planned_bytes"])

    def test_a_budget_between_the_two_plans_is_resident_but_not_the_same_run(self):
        self.assertEqual(study.expected_execution(self.PLAN, 1950), "resident")
        self.assertNotEqual(study.execution_key(self.PLAN, 1950), study.execution_key(self.PLAN, 2000))

    def test_the_log_of_an_fp32_hierarchy_reads_as_resident_with_its_plan(self):
        record = study.suite.parse_log("[cuda device AMG] fp32 hierarchy resident: 10 GPU rows, 0 CPU rows; "
                                       "fixed=100 planned=200 budget=300 bytes")
        self.assertTrue(record["fp32_hierarchy"])
        self.assertEqual((record["execution"], record["planned_bytes"], record["fixed_bytes"]), ("resident", 200, 100))
        self.assertNotIn("fp32_hierarchy", study.suite.parse_log("[cuda device AMG] 10 GPU rows, 0 CPU rows, 0 gathered inputs"))

    def test_budgets_on_one_side_of_the_fp32_plan_are_the_same_run(self):
        self.assertEqual(study.execution_key(self.PLAN, 2000), study.execution_key(self.PLAN, 5000))
        self.assertEqual(study.execution_key(self.PLAN, 1900), study.execution_key(self.PLAN, 1950))
        self.assertEqual(study.execution_key(RIGID, 1900), study.execution_key(RIGID, 5000))


class Execution(unittest.TestCase):
    def test_a_run_that_spilled_instead_of_staying_resident_is_a_problem(self):
        self.assertIn("spill", study.execution_problem({"outcome": "solved", "execution": "spill"}, "resident"))

    def test_a_solved_run_with_no_recorded_execution_is_a_problem(self):
        self.assertIsNotNone(study.execution_problem({"outcome": "solved"}, "host"))

    def test_a_failed_solve_is_an_outcome_and_not_an_execution_problem(self):
        self.assertIsNone(study.execution_problem({"outcome": "timeout"}, "resident"))


class LargeDecks(unittest.TestCase):
    """With no forced budget, the run's log holds the partition the planner chose."""

    def problem(self, **fields):
        return study.own_plan_problem({"outcome": "solved", "execution": "spill", **fields})

    def test_a_spilled_choice_that_fills_the_budget_exactly_is_fine(self):
        self.assertIsNone(self.problem(planned_bytes=8202 * MIB, budget_bytes=8202 * MIB))

    def test_a_choice_over_the_budget_is_a_problem(self):
        self.assertIn("over a budget", self.problem(planned_bytes=8202 * MIB + 1, budget_bytes=8202 * MIB))

    def test_a_solve_with_no_plan_is_a_problem_unless_it_ran_on_the_host(self):
        self.assertIn("no plan", self.problem())
        self.assertIsNone(self.problem(execution="host"))

    def test_a_failed_solve_is_an_outcome_and_not_a_plan_problem(self):
        self.assertIsNone(study.own_plan_problem({"outcome": "failed"}))


class Verdicts(unittest.TestCase):
    def test_a_range_spanning_one_is_a_tie_however_far_the_median_is(self):
        self.assertEqual(study.verdict([0.5, 0.6, 0.7, 0.8, 1.01]), "tie")

    def test_a_win_needs_every_pair(self):
        self.assertEqual(study.verdict([0.5, 0.9, 0.99]), "rigid")
        self.assertEqual(study.verdict([1.01, 3.0]), "constant")
        self.assertEqual(study.verdict([]), "unpaired")

    def test_ratios_use_only_repeats_where_both_solved(self):
        samples = {0: {"constant": 2.0, "rigid": 1.0}, 1: {"rigid": 5.0}, 2: {"constant": 1.0, "rigid": 3.0}}
        self.assertEqual(study.paired_ratios(samples), [0.5, 3.0])


class Criteria(unittest.TestCase):
    SOLVED = {"outcome": "solved", "complete_s": 10.0, "solver_s": 5.0}

    def test_failing_where_the_other_mode_solves_breaks_them(self):
        failures = study.criteria_failures({("c", "both-resident"): {"constant": {"outcome": "timeout"}, "rigid": self.SOLVED}})
        self.assertEqual(len(failures["constant"]), 1)
        self.assertEqual(failures["rigid"], [])

    def test_a_cell_neither_solves_counts_against_neither(self):
        failures = study.criteria_failures({("c", "both-host"): {"constant": {"outcome": "timeout"}, "rigid": {"outcome": "failed"}}})
        self.assertEqual(failures, {"constant": [], "rigid": []})

    def test_complete_and_solver_limits_apply_separately(self):
        slower = {"outcome": "solved", "complete_s": 19.0, "solver_s": 21.0}
        failures = study.criteria_failures({("c", "rigid-50"): {"constant": self.SOLVED, "rigid": slower}})
        self.assertEqual(len(failures["rigid"]), 1)
        self.assertIn("solver", failures["rigid"][0])
        slower = {"outcome": "solved", "complete_s": 21.0, "solver_s": 19.0}
        failures = study.criteria_failures({("c", "rigid-50"): {"constant": self.SOLVED, "rigid": slower}})
        self.assertEqual(len(failures["rigid"]), 1)
        self.assertIn("complete", failures["rigid"][0])


if __name__ == "__main__":
    unittest.main()
