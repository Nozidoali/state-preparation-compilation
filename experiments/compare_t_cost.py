"""Compare T-cost of rotation-based vs QROM-based state preparation.

Usage
-----
    python -m experiments.compare_t_cost [options]

Examples
--------
    python -m experiments.compare_t_cost --n-max 5 -v
    python -m experiments.compare_t_cost -f w --precision 4,8
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from qiskit.quantum_info import Statevector

from qsp.metrics import count_cnots, count_ry, estimate_t_count
from qsp.qrom.alias_sampling import estimate_alias_sampling_t_count, synthesize_alias_sampling
from qsp.rotation.dense import prepare_state_dense
from qsp.rotation.sparse import prepare_state_sparse
from qsp.state import (
    State,
    dense_uniform,
    dicke_state,
    random_dense_state,
    random_sparse_state,
    sparse_random,
    sparse_uniform,
    to_statevector,
    w_state,
)


@dataclass
class TcostResult:
    family: str = ""
    n: int = 0
    m: int = 0
    approach: str = ""
    precision_bits: int = 0
    t_count: int = 0
    ry_count: int = 0
    cnot_count: int = 0
    qubit_count: int = 0
    gate_count: int = 0
    time_ms: float = 0.0
    fidelity: float = 0.0
    success: bool = True
    qc: Optional[object] = None
    gs_t_count: int = 0


def generate_state(family: str, n: int, seed: int) -> tuple[State, int]:
    if family == "w":
        return w_state(n)
    if family == "dicke_2":
        return dicke_state(n, 2)
    if family == "dicke_3":
        return dicke_state(n, 3)
    if family == "dense_uniform":
        return dense_uniform(n)
    if family == "sparse_uniform":
        return sparse_uniform(n, seed)
    if family == "sparse_random":
        return sparse_random(n, seed)
    if family == "dense_random":
        return random_dense_state(n, seed)
    return {0: 1.0}, n


def _compute_fidelity(qc, target: State, n_bits: int) -> float:
    try:
        sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
        sim = np.real(sv.data[: 1 << n_bits])
        tgt = to_statevector(target, n_bits)
        return float(np.abs(np.dot(sim, tgt)) ** 2)
    except Exception:
        return 0.0


def _compute_fidelity_qrom(qc, target: State, n_bits: int) -> float:
    try:
        sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
        probs = np.zeros(1 << n_bits)
        for idx, amp in enumerate(sv.data):
            probs[idx & ((1 << n_bits) - 1)] += abs(amp) ** 2
        tgt = to_statevector(target, n_bits)
        tgt_probs = tgt ** 2
        return float(np.sum(np.sqrt(probs * tgt_probs)) ** 2)
    except Exception:
        return 0.0


# --- Gridsynth Clifford+T decomposition ---

def _gridsynth_t_count_single(theta: float, epsilon: float) -> int:
    """T-count for a single Ry(theta) via gridsynth at precision epsilon."""
    if abs(theta) < 1e-12:
        return 0
    if abs(theta - math.pi) < 1e-12 or abs(theta + math.pi) < 1e-12:
        return 0

    warnings.filterwarnings("ignore")
    from pygridsynth import gridsynth_gates
    gates = gridsynth_gates(theta, epsilon)
    return gates.count("T")


_gs_cache: Dict[tuple, int] = {}


def _gridsynth_t_count(qc, precision_bits: int) -> int:
    """Total T-count for a rotation circuit using gridsynth decomposition.

    Each Ry(theta) is decomposed to Clifford+T at epsilon = 2^{-precision_bits}
    via the Ross-Selinger gridsynth algorithm.  CCX gates cost 4T each.
    """
    epsilon = 2.0 ** (-precision_bits)
    total = 0
    for instr in qc.data:
        name = instr.operation.name
        if name == "ry":
            theta = float(instr.operation.params[0])
            key = (round(theta * 1e10), precision_bits)
            if key not in _gs_cache:
                _gs_cache[key] = _gridsynth_t_count_single(theta, epsilon)
            total += _gs_cache[key]
        elif name == "ccx":
            total += 4
        elif name in ("t", "tdg"):
            total += 1
    return total


def run_rotation(
    state: State, n_bits: int, family: str, strategy: str, timeout: float,
    precision_bits: int = 0,
) -> TcostResult:
    r = TcostResult(family=family, n=n_bits, m=len(state),
                    approach=f"rotation_{strategy}", precision_bits=precision_bits)
    start = time.monotonic()
    try:
        if strategy == "sparse":
            qc = prepare_state_sparse(state, n_bits, timeout)
        else:
            qc = prepare_state_dense(state, n_bits, timeout)
        r.time_ms = (time.monotonic() - start) * 1000.0
        r.t_count = estimate_t_count(qc)
        r.ry_count = count_ry(qc)
        r.cnot_count = count_cnots(qc)
        r.qubit_count = qc.num_qubits
        r.gate_count = qc.size()
        r.qc = qc
        if n_bits <= 10:
            r.fidelity = _compute_fidelity(qc, state, n_bits)
        if precision_bits > 0 and r.ry_count > 0:
            r.gs_t_count = _gridsynth_t_count(qc, precision_bits)
    except Exception:
        r.success = False
        r.time_ms = (time.monotonic() - start) * 1000.0
    return r


def run_qrom(
    state: State, n_bits: int, family: str, precision_bits: int, use_select_swap: bool
) -> TcostResult:
    approach = "selectswap" if use_select_swap else "qrom"
    r = TcostResult(family=family, n=n_bits, m=len(state), approach=approach,
                    precision_bits=precision_bits)
    start = time.monotonic()
    try:
        res = synthesize_alias_sampling(state, n_bits, precision_bits, use_select_swap)
        r.time_ms = res.compile_time_ms
        r.t_count = res.t_count
        r.cnot_count = res.cnot_count
        r.qubit_count = res.qubit_count
        r.gate_count = res.qc.size()
        r.qc = res.qc
        if res.qc.num_qubits <= 25:
            r.fidelity = _compute_fidelity_qrom(res.qc, state, n_bits)
    except Exception:
        r.success = False
        r.time_ms = (time.monotonic() - start) * 1000.0
    return r


CSV_HEADER = [
    "family", "n", "m", "approach", "precision_bits",
    "t_count", "ry_count", "cnot_count", "qubit_count", "gate_count",
    "time_ms", "fidelity", "success", "gs_t_count",
]


def run_benchmarks(args: argparse.Namespace) -> List[TcostResult]:
    families = (
        ["w", "dicke_2", "sparse_uniform", "sparse_random", "dense_uniform"]
        if args.family == "all"
        else [args.family]
    )
    precisions = [int(x) for x in args.precision.split(",")]
    results: List[TcostResult] = []

    for family in families:
        max_n = args.n_max
        if "dense" in family:
            max_n = min(max_n, 12)

        if args.verbose:
            print(f"\nFamily: {family} (n={args.n_min}..{max_n})", file=sys.stderr)

        for n in range(args.n_min, max_n + 1):
            is_random = "random" in family
            num_runs = args.seeds if is_random else 1

            for seed in range(1, num_runs + 1):
                state, n_bits = generate_state(family, n, seed)

                if args.verbose:
                    msg = f"  n={n}, m={len(state)}"
                    if is_random:
                        msg += f", seed={seed}"
                    print(msg, file=sys.stderr)

                for b in precisions:
                    for strat in ["sparse", "dense"]:
                        r = run_rotation(state, n_bits, family, strat, args.timeout,
                                         precision_bits=b)
                        results.append(r)
                        if args.verbose:
                            _print_result(r)

                    r_basic = run_qrom(state, n_bits, family, b, False)
                    results.append(r_basic)
                    if args.verbose:
                        _print_result(r_basic)
                    r_swap = run_qrom(state, n_bits, family, b, True)
                    results.append(r_swap)
                    if args.verbose:
                        _print_result(r_swap)

    return results


def _print_result(r: TcostResult) -> None:
    status = "" if r.success else " [FAILED]"
    fid_str = f" fid={r.fidelity:.4f}" if r.fidelity > 0 else ""
    ry_str = f" Ry={r.ry_count:4d}" if r.ry_count > 0 else ""
    gs_str = f" GS_T={r.gs_t_count:6d}" if r.gs_t_count > 0 else ""
    print(
        f"    {r.approach:16s} b={r.precision_bits:2d}"
        f" T={r.t_count:8d}{ry_str}{gs_str} CNOT={r.cnot_count:6d}"
        f" q={r.qubit_count:4d} t={r.time_ms:8.1f}ms{fid_str}{status}",
        file=sys.stderr,
    )


def write_csv(results: List[TcostResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in results:
            w.writerow([
                r.family, r.n, r.m, r.approach, r.precision_bits,
                r.t_count, r.ry_count, r.cnot_count, r.qubit_count, r.gate_count,
                f"{r.time_ms:.2f}", f"{r.fidelity:.6f}",
                "true" if r.success else "false",
                r.gs_t_count,
            ])


def save_circuits(results: List[TcostResult], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qiskit.qasm2 import dumps as qasm2_dumps

    qasm_dir = out_dir / "qasm"
    png_dir = out_dir / "png"
    qasm_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        if r.qc is None or not r.success:
            continue
        b_tag = f"_b{r.precision_bits}" if r.precision_bits > 0 else ""
        name = f"{r.family}_n{r.n}_{r.approach}{b_tag}"

        qasm_path = qasm_dir / f"{name}.qasm"
        qasm_path.write_text(qasm2_dumps(r.qc))

        png_path = png_dir / f"{name}.png"
        fig = r.qc.draw(output="mpl", fold=-1)
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        plt.close(fig)


def _print_comparison_table(results: List[TcostResult]) -> None:
    groups: Dict[tuple, List[TcostResult]] = {}
    for r in results:
        if not r.success:
            continue
        key = (r.family, r.n, r.precision_bits)
        groups.setdefault(key, []).append(r)

    print("\n=== T-count Comparison (Rotation gridsynth vs QROM) ===\n")
    header = (f"{'state':>15s} {'approach':>18s} {'b':>3s}"
              f" {'T':>8s} {'GS_T':>8s} {'Ry':>5s} {'CNOT':>6s} {'q':>4s}")
    print(header)
    print("-" * len(header))

    prev_key = None
    for key in sorted(groups.keys()):
        if prev_key and (key[0], key[1]) != (prev_key[0], prev_key[1]):
            print()
        prev_key = key
        for r in groups[key]:
            t_str = str(r.t_count) if r.t_count > 0 else "-"
            gs_str = str(r.gs_t_count) if r.gs_t_count > 0 else "-"
            ry_str = str(r.ry_count) if r.ry_count > 0 else "-"
            print(
                f"{r.family+'_'+str(r.n):>15s} {r.approach:>18s} {r.precision_bits:3d}"
                f" {t_str:>8s} {gs_str:>8s} {ry_str:>5s} {r.cnot_count:>6d}"
                f" {r.qubit_count:>4d}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="QSP T-cost comparison benchmark")
    parser.add_argument("--n-min", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("-f", "--family", default="all")
    parser.add_argument("-b", "--precision", default="1,2,4,8")
    parser.add_argument("-s", "--seeds", type=int, default=3)
    parser.add_argument("-t", "--timeout", type=float, default=120.0)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--save-circuits", action="store_true",
                        help="Save circuits as QASM and PNG")
    args = parser.parse_args()

    results = run_benchmarks(args)

    out_path = Path(args.output) if args.output else Path("qsp_t_cost_results.csv")
    write_csv(results, out_path)
    print(f"Wrote {len(results)} rows to {out_path}")

    if args.save_circuits:
        circ_dir = out_path.parent / "circuits"
        save_circuits(results, circ_dir)
        print(f"Saved circuits to {circ_dir}")

    _print_comparison_table(results)


if __name__ == "__main__":
    main()
