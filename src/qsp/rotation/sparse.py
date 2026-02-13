"""Cardinality reduction strategy for state preparation.

Iteratively reduces the number of nonzero amplitudes (cardinality) by one
at each step, using controlled rotations that merge pairs of basis states.

Ported from ``strategy-sparse.cpp``.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Set, Tuple

from qiskit.circuit import QuantumCircuit

from qsp.state import EPS, State, cardinality, state_hash
from qsp.rotation.common import (
    apply_controlled_cx,
    apply_mcry,
    apply_x,
    maximize_difference_once,
)


def cardinality_reduction_by_one(
    state: State, n_bits: int
) -> Tuple[State, List[dict]]:
    new_state = dict(state)
    indices: Set[int] = set(state.keys())

    diff_values: Dict[int, bool] = {}
    diff_qubit = 0
    while len(indices) > 1:
        diff_qubit, _ = maximize_difference_once(n_bits, indices, diff_values)

    index0 = next(iter(indices))
    diff_values.pop(diff_qubit, None)

    candidates: Set[int] = set()
    for idx in state:
        if idx in indices:
            continue
        valid = all(
            bool((idx >> q) & 1) == v for q, v in diff_values.items()
        )
        if valid:
            candidates.add(idx)
    while len(candidates) > 1:
        maximize_difference_once(n_bits, candidates, diff_values)

    index1 = next(iter(candidates))
    gates: List[dict] = []

    diff_val = bool((index0 >> diff_qubit) & 1)

    for qubit in range(n_bits):
        if ((index0 >> qubit) & 1) == ((index1 >> qubit) & 1):
            continue
        if qubit == diff_qubit:
            continue
        gate = {"type": "cx", "control": diff_qubit, "ctrl_val": diff_val, "target": qubit}
        gates.append(gate)
        new_state = apply_controlled_cx(new_state, diff_qubit, diff_val, qubit)

    ctrls = []
    phases = []
    for q, v in diff_values.items():
        ctrls.append(q)
        phases.append(v)

    idx0 = index1 & ~(1 << diff_qubit)
    idx1 = index1 | (1 << diff_qubit)
    assert idx0 in new_state, f"idx0={idx0} not in state"
    assert idx1 in new_state, f"idx1={idx1} not in state"

    w0 = new_state[idx0]
    w1 = new_state[idx1]
    theta = 2.0 * math.atan2(w0, w1)
    if (index1 >> diff_qubit) & 1:
        theta = -math.pi + theta

    gate = {"type": "mcry", "ctrls": ctrls, "phases": phases,
            "theta": theta, "target": diff_qubit}
    gates.append(gate)
    new_state = apply_mcry(new_state, ctrls, phases, theta, diff_qubit)

    return new_state, gates


def prepare_state_sparse(
    state: State,
    n_bits: int,
    timeout: float = 120.0,
) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits)
    curr = dict(state)
    all_gates: List[dict] = []

    max_iter = 1000
    seen: set = set()
    no_reduction = 0
    prev_card = cardinality(curr)
    start = time.monotonic()

    for _ in range(max_iter):
        if cardinality(curr) <= 1:
            break
        if time.monotonic() - start >= timeout:
            break

        h = state_hash(curr, n_bits)
        if h in seen:
            break
        seen.add(h)

        new_state, gates = cardinality_reduction_by_one(curr, n_bits)
        new_card = cardinality(new_state)
        if new_card >= prev_card:
            no_reduction += 1
            if no_reduction > 3:
                break
        else:
            all_gates.extend(gates)
            curr = new_state
            no_reduction = 0
        prev_card = cardinality(curr)

    if curr:
        index = next(iter(curr.keys()))
        for qubit in range(n_bits):
            if (index >> qubit) & 1:
                all_gates.append({"type": "x", "target": qubit})

    all_gates.reverse()
    _emit_gates(qc, all_gates)
    return qc


def _emit_gates(qc: QuantumCircuit, gates: List[dict]) -> None:
    for g in gates:
        if g["type"] == "x":
            qc.x(g["target"])
        elif g["type"] == "cx":
            ctrl = g["control"]
            target = g["target"]
            ctrl_val = g["ctrl_val"]
            if not ctrl_val:
                qc.x(ctrl)
            qc.cx(ctrl, target)
            if not ctrl_val:
                qc.x(ctrl)
        elif g["type"] == "mcry":
            theta = -g["theta"]
            ctrls = g["ctrls"]
            phases = g["phases"]
            target = g["target"]
            if not ctrls:
                qc.ry(theta, target)
            else:
                for i, p in enumerate(phases):
                    if not p:
                        qc.x(ctrls[i])
                if len(ctrls) == 1:
                    qc.cry(theta, ctrls[0], target)
                else:
                    from qiskit.circuit.library import RYGate
                    gate = RYGate(theta).control(len(ctrls))
                    qc.append(gate, ctrls + [target])
                for i, p in enumerate(phases):
                    if not p:
                        qc.x(ctrls[i])
