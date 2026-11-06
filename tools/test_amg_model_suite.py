#!/usr/bin/env python3
"""The suite's acceptance checks reject what they exist to reject.

A benchmark that cannot fail proves nothing, so each check here is given the wrong
answer it guards against, through a small fake solver where the process status
matters.
"""
import io
import json
import math
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import amg_model_suite as suite

# A 2-equation system with a known solution, in the formats --dump-system writes.
K = "%%MatrixMarket matrix coordinate real symmetric\n%\n2 2 3\n1 1 2\n2 1 -1\n2 2 2\n"
F = "%%MatrixMarket matrix array real general\n%\n2 1\n1\n1\n"
DOFS = "# equation grid component\n1 7 1\n2 7 4\n"
F06 = """\
                  D I S P L A C E M E N T S
      GRID     COORD      T1            T2            T3            R1            R2            R3
         7        0  {u:.6E}  0.0           0.0           {u:.6E}  0.0           0.0
                  S P C   F O R C E S
      GRID     COORD      T1            T2            T3            R1            R2            R3
         8        0  {r:.6E}  0.0           0.0           0.0           0.0           0.0
"""


def write_system(directory, u, k=K):
    prefix = Path(directory) / "x"
    (directory / "x_sc1_K.mtx").write_text(k)
    (directory / "x_sc1_F.mtx").write_text(F)
    (directory / "x_sc1_u.mtx").write_text("%%MatrixMarket matrix array real general\n2 1\n" + "".join(f"{v:.17g}\n" for v in u))
    (directory / "x_sc1_dofs.txt").write_text(DOFS)
    return prefix


class VerifyAgainst(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.reference = suite.load_system(write_system(self.dir, [1.0, 1.0]), True)
        self.reactions = {(0, 8): [1.0, 0, 0, 0, 0, 0]}

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self, u, k=K, reactions=None):
        system = suite.load_system(write_system(self.dir, u, k), False)
        return suite.verify_against(system, self.reference, self.reactions if reactions is None else reactions, self.reactions)

    def test_the_exact_solution_passes_with_every_measure_recorded(self):
        failures, measures = self.run_check([1.0, 1.0])
        self.assertEqual(failures, [])
        self.assertEqual(set(measures), set(suite.TOLERANCES))
        self.assertTrue(all(v == 0 for v in measures.values()), measures)

    def test_translations_and_rotations_are_judged_separately(self):
        # Here a rotation error would vanish in a norm over a much larger translation.
        reference = suite.load_system(write_system(self.dir, [1e6, 1.0]), True)
        system = suite.load_system(write_system(self.dir, [1e6, 1.0 + 1e-3]), False)
        failures, _ = suite.verify_against(system, reference, self.reactions, self.reactions)
        self.assertTrue(any(f.startswith("rotation") for f in failures), failures)
        self.assertFalse(any(f.startswith("translation") for f in failures), failures)

    def test_the_residual_limit_rises_only_to_what_the_reference_itself_attains(self):
        # Reference residual 1e-4: its floor of 1e-3 admits a residual of 5e-4, not 2e-2.
        reference = suite.load_system(write_system(self.dir, [1.0 + 1e-4, 1.0 + 1e-4]), True)
        near = suite.load_system(write_system(self.dir, [1.0 + 1e-4 + 2e-4, 1.0 + 1e-4 + 2e-4]), False)
        far = suite.load_system(write_system(self.dir, [1.0 + 1e-2, 1.0 + 1e-2]), False)
        self.assertFalse(any(f.startswith("residual") for f in suite.verify_against(near, reference, self.reactions, self.reactions)[0]))
        self.assertTrue(any(f.startswith("residual") for f in suite.verify_against(far, reference, self.reactions, self.reactions)[0]))

    def test_a_wrong_solution_fails_residual_and_energy(self):
        failures, _ = self.run_check([1.0 + 1e-3, 1.0 - 1e-3])
        for name in ("translation", "rotation", "residual", "energy"):
            self.assertTrue(any(f.startswith(name) for f in failures), (name, failures))

    def test_non_finite_values_fail(self):
        failures, _ = self.run_check([float("nan"), 1.0])
        self.assertTrue(any("non-finite" in f for f in failures), failures)

    def test_a_different_assembled_matrix_fails(self):
        failures, _ = self.run_check([1.0, 1.0], K.replace("1 1 2", "1 1 2.5"))
        self.assertTrue(any("assembled K" in f for f in failures), failures)

    def test_missing_equations_and_subcases_fail(self):
        system = suite.load_system(write_system(self.dir, [1.0, 1.0]), False)
        system["1"]["dofs"] = system["1"]["dofs"][:1]
        self.assertTrue(suite.verify_against(system, self.reference, self.reactions, self.reactions)[0])
        self.assertTrue(suite.verify_against({}, self.reference, self.reactions, self.reactions)[0])

    def test_incomplete_or_wrong_reactions_fail(self):
        self.assertTrue(self.run_check([1.0, 1.0], reactions={})[0])
        failures, _ = self.run_check([1.0, 1.0], reactions={(0, 8): [1.001, 0, 0, 0, 0, 0]})
        self.assertTrue(any(f.startswith("reaction") for f in failures), failures)

    def test_a_reference_without_reactions_cannot_pass(self):
        system = suite.load_system(write_system(self.dir, [1.0, 1.0]), False)
        self.assertTrue(suite.verify_against(system, self.reference, {}, {})[0])

    def test_partial_tables_do_not_compare_only_what_overlaps(self):
        self.assertEqual(suite.relative_difference({(0, 1): [1.0]}, {(0, 1): [1.0], (0, 2): [1.0]}), float("inf"))


# Answers as a solver would: --dump-system files, an F06 and a log. The answer is
# taken from an environment-free file next to it, per backend.
FAKE = r'''#!/usr/bin/env python3
import sys, re
args = sys.argv[1:]
option = lambda name: next((a.split("=", 1)[1] for a in args if a.startswith(name + "=")), None)
behaviour = open(__file__ + ".behaviour").read().strip()
pcg = option("--backend") != "cpu"
if pcg and behaviour == "crash": sys.exit(3)
if not pcg: assert "--refine" in args, "the reference must be refined"
u = 1.0 + (1e-3 if pcg and behaviour == "wrong" else 0.0)
positional = [a for a in args if not a.startswith("--")]
open(option("--log-file"), "w").write(
    "[amg-pcg] n=2, nnz=4, setup=1.0 ms, iteration/refinement=2.0 ms, iterations=7, estimated error=1e-9\n"
    + ("[amg-pcg] cycle probe: valid=false asymmetry=1e-3 min_energy=-1 max_energy=2 contraction=3 last_contraction=3\n" if pcg and behaviour == "indefinite" else ""))
table = open(__file__ + ".f06").read().format(u=u, r=1.0)
open(positional[1], "w").write(table)
prefix = option("--dump-system")
if prefix:
    open(prefix + "_sc1_K.mtx", "w").write(open(__file__ + ".k").read())
    open(prefix + "_sc1_F.mtx", "w").write(open(__file__ + ".f").read())
    open(prefix + "_sc1_u.mtx", "w").write("%%MatrixMarket matrix array real general\n2 1\n" + f"{u:.17g}\n{u:.17g}\n")
    open(prefix + "_sc1_dofs.txt", "w").write(open(__file__ + ".dofs").read())
'''


class ProcessStatus(unittest.TestCase):
    def run_suite(self, behaviour, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fake = tmp / "fake"
            fake.write_text(FAKE)
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            (tmp / "fake.behaviour").write_text(behaviour)
            (tmp / "fake.f06").write_text(F06)
            (tmp / "fake.k").write_text(K)
            (tmp / "fake.f").write_text(F)
            (tmp / "fake.dofs").write_text(DOFS)
            results = tmp / "r.jsonl"
            done = subprocess.run(
                [sys.executable, str(Path(suite.__file__)), "run", "--binary", str(fake), "--work", str(tmp / "w"),
                 "--results", str(results), "--modes", "rigid", "--classes", "shell-thick", "--levels", "2",
                 "--direct", *extra], capture_output=True, text=True)
            rows = [json.loads(line) for line in results.read_text().splitlines()]
            return done, rows

    def test_a_correct_run_exits_zero_with_its_record(self):
        done, rows = self.run_suite("right", "--require-solved", "rigid")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual([r["modes"] for r in rows], ["direct", "rigid"])
        self.assertEqual(rows[1]["comparison"]["failures"], [])
        self.assertEqual(rows[1]["iterations"], [7])

    def test_a_failed_comparison_exits_nonzero(self):
        done, rows = self.run_suite("wrong")
        self.assertEqual(done.returncode, 1)
        self.assertIn("translation", done.stderr)
        self.assertTrue(rows[1]["comparison"]["failures"])

    def test_a_failed_solve_is_a_recorded_outcome_unless_required(self):
        done, rows = self.run_suite("crash")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(rows[1]["outcome"], "failed")
        self.assertEqual(self.run_suite("crash", "--require-solved", "rigid")[0].returncode, 1)

    def test_an_invalid_cycle_fails_a_required_mode(self):
        self.assertEqual(self.run_suite("indefinite", "--require-solved", "rigid")[0].returncode, 1)


class TubeChassis(unittest.TestCase):
    """The deck keeps the properties that make it a chassis and not another lattice."""

    def setUp(self):
        self.joints, self.tubes, self.corners, self.supports = suite.chassis_layout()

    def test_only_cross_members_lie_on_an_axis(self):
        on_axis = [(a, b) for a, b, _ in self.tubes
                   if sum(abs(q - p) > 1e-9 for p, q in zip(self.joints[a], self.joints[b])) == 1]
        for a, b in on_axis:
            self.assertGreater(abs(self.joints[a][1] - self.joints[b][1]), 0, (a, b))
        self.assertLess(len(on_axis) / len(self.tubes), 0.25)

    def test_the_frame_is_one_structure_with_uneven_connectivity(self):
        neighbours = {name: set() for name in self.joints}
        for a, b, _ in self.tubes:
            neighbours[a].add(b)
            neighbours[b].add(a)
        seen, stack = {self.tubes[0][0]}, [self.tubes[0][0]]
        while stack:
            for name in neighbours[stack.pop()] - seen:
                seen.add(name)
                stack.append(name)
        self.assertEqual(seen, set(self.joints))
        valence = {name: sum(name in tube[:2] for tube in self.tubes) for name in self.joints}
        self.assertLessEqual(min(valence.values()), 2)
        self.assertGreaterEqual(max(valence.values()), 9)

    def test_each_corner_holds_its_upright_with_no_mechanism(self):
        for corner in self.corners:
            matrix = []
            for inner, name, _ in corner["links"]:
                a, b = self.joints[inner], corner["outer"][name]
                length = math.dist(a, b)
                d = [(q - p) / length for p, q in zip(a, b)]
                r = [q - p for p, q in zip(corner["patch"], b)]
                moment = [r[1] * d[2] - r[2] * d[1], r[2] * d[0] - r[0] * d[2], r[0] * d[1] - r[1] * d[0]]
                matrix.append(d + moment)
            # Gaussian elimination: a mechanism shows as a vanishing pivot.
            smallest = math.inf
            for column in range(6):
                pivot = max(range(column, 6), key=lambda row: abs(matrix[row][column]))
                matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
                smallest = min(smallest, abs(matrix[column][column]))
                for row in range(column + 1, 6):
                    factor = matrix[row][column] / matrix[column][column]
                    matrix[row] = [x - factor * y for x, y in zip(matrix[row], matrix[column])]
            self.assertGreater(smallest, 1e-3, (corner["end"], corner["side"]))

    def test_the_supports_are_statically_determinate(self):
        self.assertEqual(sorted(len(str(components)) for _, components in self.supports), [1, 2, 3])

    def test_refinement_splits_tubes_without_moving_them(self):
        per_metre = 40
        out = io.StringIO()
        suite.tube_chassis(out, per_metre)
        grids, frame_length, longest, used, forces = {}, 0.0, 0.0, set(), [0.0, 0.0, 0.0]
        lines = out.getvalue().splitlines()
        for line in lines:
            f = line.split(",")
            if f[0] == "GRID":
                grids[int(f[1])] = tuple(float(v) for v in f[3:6])
        for line in lines:
            f = line.split(",")
            if f[0] == "CBAR":
                used.update((int(f[3]), int(f[4])))
                if f[2] not in ("4", "5"):
                    length = math.dist(grids[int(f[3])], grids[int(f[4])])
                    frame_length += length
                    longest = max(longest, length)
            elif f[0] == "RBE2":
                used.update(int(v) for v in f[2:3] + f[4:])
            elif f[0] == "FORCE":
                for i in range(3):
                    forces[i] += float(f[4]) * float(f[5 + i])
        self.assertAlmostEqual(frame_length, sum(math.dist(self.joints[a], self.joints[b]) for a, b, _ in self.tubes), places=6)
        self.assertLessEqual(longest, 1 / per_metre + 1e-9)
        self.assertEqual(used, set(grids), "every grid is on an element")
        for i in range(3):
            self.assertAlmostEqual(forces[i], sum(v[i] for v in suite.CHASSIS_WHEEL_FORCES.values()), places=4)


if __name__ == "__main__":
    unittest.main()
