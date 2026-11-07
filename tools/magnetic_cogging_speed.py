#!/usr/bin/env python3
"""Open-circuit cogging and eddy-current braking versus signed RPM."""

import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys
import time

if sys.platform != "win32":
    import resource

import h5py
import numpy as np

from magnetic_motor_example import MAGNET_IN, MAGNET_OUT, POLE_PAIRS, SLOTS, build_mesh, write_deck


def run(solver, deck):
    start = time.perf_counter()
    result = deck.with_suffix(".mag.vtkhdf")
    result.unlink(missing_ok=True)
    with deck.with_suffix(".log").open("w") as log:
        status = subprocess.run([solver, deck], stdout=log, stderr=subprocess.STDOUT).returncode
    return result, status, time.perf_counter() - start


def harmonic(angle, torque, order, floor):
    coefficient = 2 * np.mean(torque * np.exp(-1j * order * np.deg2rad(angle)))
    amplitude = abs(coefficient)
    return amplitude, np.angle(coefficient, deg=True) if amplitude > floor else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rpm", type=float, nargs="+", required=True)
    parser.add_argument("--vibestran", type=Path, default=Path("build/vibestran"))
    parser.add_argument("--sigma", type=float, default=6e5, help="magnet conductivity in S/m")
    parser.add_argument("--mesh-scale", type=float, default=1.)
    parser.add_argument("--samples-per-cogging", type=int, default=24)
    parser.add_argument("--harmonics", type=int, help="number of cogging orders to report, up to 3 by default")
    parser.add_argument("--max-cycles", type=int, default=12)
    parser.add_argument("--fields-every", type=int, default=0, help="0 writes scalar results only")
    parser.add_argument("--phase-floor", type=float, default=1e-12, help="minimum amplitude in Nm")
    args = parser.parse_args()
    if (any(not math.isfinite(v) or v == 0 for v in args.rpm)
            or not all(math.isfinite(v) and v > 0 for v in (args.sigma, args.mesh_scale))
            or not math.isfinite(args.phase_floor) or args.phase_floor < 0
            or args.samples_per_cogging < 4 or args.max_cycles < 2 or args.fields_every < 0):
        parser.error("use finite nonzero RPM, positive conductivity and mesh scale, "
                     "at least four samples per cogging period and two revolutions")
    maximum_harmonics = (args.samples_per_cogging - 1) // 2
    harmonics = min(3, maximum_harmonics) if args.harmonics is None else args.harmonics
    if not 1 <= harmonics <= maximum_harmonics:
        parser.error(f"--harmonics must be between 1 and {maximum_harmonics} at this angular sampling")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    solver = args.vibestran.resolve()
    order = math.lcm(SLOTS, 2 * POLE_PAIRS)
    count = order * args.samples_per_cogging
    orders = [order * n for n in range(1, harmonics + 1)]
    nodes, elements = build_mesh(args.mesh_scale)
    baseline = output / "cogging_static.bdf"
    write_deck(baseline, nodes, elements, cogging_samples=count)
    print(f"Static reference on {len(elements)} triangles and {count} angles", flush=True)
    static_file, status, seconds = run(solver, baseline)
    if status:
        print(f"Static reference failed, see {baseline.with_suffix('.log')}", file=sys.stderr)
        return status
    with h5py.File(static_file, "r") as file:
        static = file["MAGSTAT/subcases/1"]
        reference_angle = static["positions_deg"][:]
        reference_torque = static["motion/torque_Nm"][:]
    print(f"Static reference took {seconds:.2f} s", flush=True)
    bulk = baseline.read_text().partition("BEGIN BULK\n")[2].splitlines()
    bulk = [line for line in bulk if not line.startswith(("MOTION,", "ENDDATA"))]
    sample = f"BFIELD(EVERY={args.fields_every}) = ALL" if args.fields_every else "BFIELD = NONE"
    loss_pids = [MAGNET_OUT, MAGNET_IN]
    columns = ["rpm", "status", "order", "order_Hz", "mean_torque_Nm", "peak_to_peak_Nm",
               "order_amplitude_Nm", "order_phase_deg", "order_amplitude_change_Nm",
               "order_phase_shift_deg", "mean_eddy_loss_W",
               "power_residual_W", "dt_s", "cycles", "symbolic_analyses",
               "numeric_factorizations", "wall_s", *(f"loss_pid_{p}_W" for p in loss_pids)]
    failed = False
    with (output / "cogging_speed.csv").open("w", newline="") as report, \
            (output / "cogging_waveforms.csv").open("w", newline="") as waveforms, \
            (output / "cogging_orders.csv").open("w", newline="") as orders_file:
        summary = csv.DictWriter(report, fieldnames=columns)
        summary.writeheader()
        waveform = csv.writer(waveforms)
        waveform.writerow(["rpm", "angle_deg", "torque_Nm", "static_torque_Nm", "delta_torque_Nm"])
        spectrum = csv.writer(orders_file)
        spectrum.writerow(["rpm", "order", "frequency_Hz", "amplitude_Nm", "phase_deg",
                           "amplitude_change_Nm", "phase_shift_deg"])
        static_spectrum = [harmonic(reference_angle, reference_torque, n, args.phase_floor)
                           for n in orders]
        static_amplitude, static_phase = static_spectrum[0]
        for n, (amplitude, phase) in zip(orders, static_spectrum):
            spectrum.writerow([0, n, 0, amplitude, phase, "", ""])
        summary.writerow(dict(rpm=0, status="static", order=order, order_Hz=0,
                              mean_torque_Nm=np.mean(reference_torque),
                              peak_to_peak_Nm=np.ptp(reference_torque),
                              order_amplitude_Nm=static_amplitude,
                              order_phase_deg=static_phase, wall_s=seconds))
        for index, rpm in enumerate(args.rpm, 1):
            deck = output / f"speed_{index:03d}.bdf"
            maximum_dt = math.nextafter(60 / abs(rpm) / count, math.inf)
            deck.write_text("\n".join([
                "SOL MAGTRANS", "CEND", "SUBCASE 1", "  MOTION = 2", "  TSTEP = 1", "  " + sample,
                "BEGIN BULK", *bulk, "MOTION,2,20,0,ROTATE,0.", f",{rpm:.17g}",
                f"TSTEP,1,{maximum_dt:.17g},{args.max_cycles}", f"MATCOND,3,{args.sigma:.17g}",
                "ENDDATA", "",
            ]))
            print(f"Solving at {rpm:g} rpm", flush=True)
            path, status, seconds = run(solver, deck)
            row = dict(rpm=rpm, status=f"failed ({status})", order=order,
                       order_Hz=order * abs(rpm) / 60, wall_s=seconds)
            if status in (0, 5) and path.is_file():
                with h5py.File(path, "r") as file:
                    case = file["MAGTRANS/subcases/1"]
                    attrs = case.attrs
                    for key in ("dt_s", "cycles", "symbolic_analyses", "numeric_factorizations"):
                        row[key] = attrs[key]
                    if status == 5 and not attrs["converged"]:
                        row["status"] = "unsettled or interrupted"
                    if status == 0 and attrs["converged"]:
                        row["status"] = "converged"
                        window = slice(-int(attrs["steps_per_cycle"]), None)
                        angle = np.mod(case["angle_deg"][window], 360.)
                        torque = case["torque_Nm"][window]
                        reference = np.interp(angle, reference_angle, reference_torque, period=360)
                        for k in np.argsort(angle):
                            waveform.writerow([rpm, angle[k], torque[k], reference[k],
                                               torque[k] - reference[k]])
                        for key in ("mean_torque_Nm", "mean_eddy_loss_W", "power_residual_W"):
                            row[key] = attrs[key]
                        row["peak_to_peak_Nm"] = np.ptp(torque)
                        for n, (base_amplitude, base_phase) in zip(orders, static_spectrum):
                            amplitude, phase = harmonic(angle, torque, n, args.phase_floor)
                            change = amplitude - base_amplitude
                            shift = (math.remainder(phase - base_phase, 360)
                                     if phase != "" and base_phase != "" else "")
                            spectrum.writerow([rpm, n, n * abs(rpm) / 60,
                                               amplitude, phase, change, shift])
                            if n == order:
                                row.update(order_amplitude_Nm=amplitude, order_phase_deg=phase,
                                           order_amplitude_change_Nm=change, order_phase_shift_deg=shift)
                        for pid, loss in zip(case["property_id"][:], case["mean_property_loss_W"][:]):
                            row[f"loss_pid_{pid}_W"] = loss
            failed |= row["status"] != "converged"
            summary.writerow(row)
            report.flush()
            print(f"  {row['status']} in {seconds:.2f} s, log {deck.with_suffix('.log')}", flush=True)
    if sys.platform != "win32":
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        print(f"Largest solver peak RSS {peak_bytes / 2**20:.1f} MiB")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
