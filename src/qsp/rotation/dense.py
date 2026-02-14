"""Qubit reduction strategy for state preparation.

Iteratively reduces the number of support qubits by one at each step,
using uniformly controlled rotations. Falls back to sparse strategy
if stuck.

Ported from ``strategy-dense.cpp``.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from qiskit.circuit import QuantumCircuit

from qsp.state import EPS, State, cardinality, get_supports, state_hash
from qsp.rotation.common import (
    apply_ry,
    apply_ucry,
    apply_x,
    apply_cx,
    get_ap_ry_angle,
    get_const1_signature,
    get_qubit_signatures,
    rotation_angles_optimization,
    select_informative_qubit,
)
from qsp.rotation.sparse import prepare_state_sparse


# --- Support reduction helpers (X/CNOT cleanup + RY) ---

def _x_reduction(state: State, n_bits: int, enable_cnot: bool) -> Tuple[State, List[dict]]:
    sigs = get_qubit_signatures(state, n_bits)
    const1 = get_const1_signature(state)
    sig_to_qubit: Dict[int, int] = {}
    curr = dict(state)
    gates: List[dict] = []

    for qi in range(n_bits):
        sig = sigs[qi]
        if sig == 0:
            continue
        if sig == const1:
            gates.append({"type": "x", "target": qi})
            curr = apply_x(curr, qi)
            continue
        if enable_cnot and sig in sig_to_qubit:
            ctrl = sig_to_qubit[sig]
            gates.append({"type": "cx_plain", "control": ctrl, "target": qi, "ctrl_val": True})
            curr = apply_cx(curr, ctrl, qi)
            sig_to_qubit[sig] = qi
            continue
        if enable_cnot and (sig ^ const1) in sig_to_qubit:
            ctrl = sig_to_qubit[sig ^ const1]
            gates.append({"type": "cx_plain", "control": ctrl, "target": qi, "ctrl_val": False})
            new: State = {}
            for idx, amp in curr.items():
                nidx = idx
                if not ((idx >> ctrl) & 1):
                    nidx ^= (1 << qi)
                new[nidx] = amp
            curr = new
            continue
        if enable_cnot:
            sig_to_qubit[sig] = qi
        else:
            sig_to_qubit[sig] = qi

    return curr, gates


def _ry_reduction(state: State, n_bits: int) -> Tuple[State, List[dict]]:
    curr = dict(state)
    gates: List[dict] = []
    for qi in range(n_bits):
        theta = get_ap_ry_angle(curr, n_bits, qi)
        if theta is not None:
            gates.append({"type": "ry", "target": qi, "theta": theta})
            curr = apply_ry(curr, qi, theta)
    return curr, gates


def _support_reduction(state: State, n_bits: int) -> Tuple[State, List[dict]]:
    s1, g1 = _x_reduction(state, n_bits, True)
    s2, g2 = _ry_reduction(s1, n_bits)
    return s2, g1 + g2


def qubit_reduction_by_one(
    state: State, n_bits: int
) -> Tuple[State, List[dict]]:
    supports = get_supports(state, n_bits)
    if len(supports) <= 1 or cardinality(state) < 2:
        return state, []

    pivot = select_informative_qubit(state, n_bits, supports)

    control_indices = [s for s in supports if s != pivot]
    n_ctrls = len(control_indices)
    rot_table = [[0.0, 0.0] for _ in range(1 << n_ctrls)]

    for idx, weight in state.items():
        rot_idx = 0
        for i, q in enumerate(control_indices):
            if (idx >> q) & 1:
                rot_idx |= 1 << i
        if (idx >> pivot) & 1:
            rot_table[rot_idx][1] += weight
        else:
            rot_table[rot_idx][0] += weight

    angles: List[float] = []
    for w0, w1 in rot_table:
        if abs(w0) < EPS:
            angles.append(math.pi)
        elif abs(w1) < EPS:
            angles.append(0.0)
        else:
            angles.append(2.0 * math.atan2(w1, w0))

    opt_angles, opt_ctrls = rotation_angles_optimization(angles, control_indices)

    gates: List[dict] = []
    if len(opt_angles) == 1:
        gates.append({"type": "ry", "target": pivot, "theta": opt_angles[0]})
    else:
        gates.append({"type": "ucry", "controls": opt_ctrls,
                       "angles": opt_angles, "target": pivot})

    reduced: State = {}
    for idx, weight in state.items():
        idx0 = idx & ~(1 << pivot)
        idx1 = idx | (1 << pivot)
        w0 = state.get(idx0, 0.0)
        w1 = state.get(idx1, 0.0)
        if idx0 not in reduced:
            reduced[idx0] = math.sqrt(w0 * w0 + w1 * w1)

    reduced = {k: v for k, v in reduced.items() if abs(v) > EPS}
    return reduced, gates


def prepare_state_dense(
    state: State,
    n_bits: int,
    timeout: float = 120.0,
) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits)
    curr = dict(state)
    all_gates: List[dict] = []

    max_iter = 1000
    seen: set = set()
    start = time.monotonic()
    supports = get_supports(curr, n_bits)

    for _ in range(max_iter):
        if len(supports) <= 1:
            break
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            break

        h = state_hash(curr, n_bits)
        if h in seen:
            break
        seen.add(h)

        new_state, gates = qubit_reduction_by_one(curr, n_bits)
        all_gates.extend(gates)
        curr = new_state
        supports = get_supports(curr, n_bits)

    if cardinality(curr) == 1:
        index = next(iter(curr.keys()))
        if index != 0 or abs(curr[index] - 1.0) > 1e-6:
            for target in range(n_bits):
                if (index >> target) & 1:
                    all_gates.append({"type": "x", "target": target})
    elif cardinality(curr) > 1:
        sparse_qc = prepare_state_sparse(curr, n_bits, timeout - (time.monotonic() - start))
        all_gates.reverse()
        qc.compose(sparse_qc, inplace=True)
        _emit_gates(qc, all_gates)
        return qc

    all_gates.reverse()
    _emit_gates(qc, all_gates)
    return qc


_DEMUX_TOL = 1e-10


def _demux_ry(qc: QuantumCircuit, angles: List[float],
              target: int, controls: List[int]) -> None:
    """Decompose uniformly-controlled Ry using recursive Gray code splitting.

    Ported from hlqcs ``csd.cpp::demux_ry``.  Produces 2^k single-qubit Ry
    gates and 2^k CNOT gates for k control qubits — no unitary matrices.
    """
    if not controls:
        if abs(angles[0]) > _DEMUX_TOL:
            qc.ry(angles[0], target)
        return

    k = len(controls)
    n_angles = 1 << k
    assert len(angles) == n_angles

    if k == 1:
        a = (angles[0] + angles[1]) / 2.0
        b = (angles[0] - angles[1]) / 2.0
        if abs(a) > _DEMUX_TOL:
            qc.ry(a, target)
        qc.cx(controls[0], target)
        if abs(b) > _DEMUX_TOL:
            qc.ry(b, target)
        qc.cx(controls[0], target)
        return

    half = n_angles // 2
    even = [(angles[i] + angles[i + half]) / 2.0 for i in range(half)]
    odd = [(angles[i] - angles[i + half]) / 2.0 for i in range(half)]

    sub_controls = controls[:-1]
    _demux_ry(qc, even, target, sub_controls)
    qc.cx(controls[-1], target)
    _demux_ry(qc, odd, target, sub_controls)
    qc.cx(controls[-1], target)


def _emit_gates(qc: QuantumCircuit, gates: List[dict]) -> None:
    for g in gates:
        if g["type"] == "x":
            qc.x(g["target"])
        elif g["type"] == "ry":
            qc.ry(g["theta"], g["target"])
        elif g["type"] == "cx_plain":
            ctrl = g["control"]
            target = g["target"]
            if not g["ctrl_val"]:
                qc.x(ctrl)
            qc.cx(ctrl, target)
            if not g["ctrl_val"]:
                qc.x(ctrl)
        elif g["type"] == "ucry":
            controls = g["controls"]
            angles = g["angles"]
            target = g["target"]
            _demux_ry(qc, angles, target, controls)
        elif g["type"] == "mcry":
            from qsp.rotation.sparse import _emit_gates as _sparse_emit
            _sparse_emit(qc, [g])
