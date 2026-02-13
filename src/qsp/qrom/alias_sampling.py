"""Alias sampling state preparation via QROM.

Walker's algorithm builds an alias table from a probability distribution.
The quantum circuit uses Hadamard superposition, QROM data loads, a comparator,
and controlled-swap to prepare the target state.

Ported from ``alias-sampling.cpp``.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List

from qiskit.circuit import QuantumCircuit

from qsp.circuit_utils import add_comparator, add_controlled_swap
from qsp.qrom.synthesis import (
    QROMResult,
    compute_optimal_lambda,
    select_qrom_t_count,
    select_swap_qrom_t_count,
    synthesize_qrom,
)
from qsp.state import State


@dataclass
class AliasTable:
    keep_values: List[int] = field(default_factory=list)
    alias_indices: List[int] = field(default_factory=list)
    n_bins: int = 0
    precision_bits: int = 0


@dataclass
class AliasSamplingResult:
    qc: QuantumCircuit = field(default_factory=lambda: QuantumCircuit(0))
    t_count: int = 0
    cnot_count: int = 0
    qubit_count: int = 0
    keep_qrom_t: int = 0
    alias_qrom_t: int = 0
    compare_t: int = 0
    cswap_t: int = 0
    compile_time_ms: float = 0.0


def _amplitudes_to_probabilities(state: State, n_bits: int) -> List[float]:
    dim = 1 << n_bits
    probs = [0.0] * dim
    for idx, amp in state.items():
        probs[idx] = amp * amp
    return probs


def build_alias_table(state: State, n_bits: int, precision_bits: int) -> AliasTable:
    probs = _amplitudes_to_probabilities(state, n_bits)
    N = len(probs)
    max_val = (1 << precision_bits) - 1

    scaled = [p * N for p in probs]

    small: deque = deque()
    large: deque = deque()
    for i in range(N):
        if scaled[i] < 1.0:
            small.append(i)
        else:
            large.append(i)

    keep_prob = [1.0] * N
    alias = [0] * N

    while small and large:
        s = small.popleft()
        l = large.popleft()
        keep_prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = (scaled[l] + scaled[s]) - 1.0
        if scaled[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    while large:
        keep_prob[large.popleft()] = 1.0
    while small:
        keep_prob[small.popleft()] = 1.0

    table = AliasTable(n_bins=N, precision_bits=precision_bits)
    for i in range(N):
        clamped = max(0.0, min(1.0, keep_prob[i]))
        table.keep_values.append(int(round(clamped * max_val)))
        table.alias_indices.append(alias[i])

    return table


def synthesize_alias_sampling(
    state: State,
    n_bits: int,
    precision_bits: int = 8,
    use_select_swap: bool = True,
    clean_ancilla: bool = True,
) -> AliasSamplingResult:
    start = time.monotonic()

    n = n_bits
    b = precision_bits

    table = build_alias_table(state, n, b)

    keep_res = synthesize_qrom(
        table.keep_values, n, b,
        use_select_swap=use_select_swap,
        clean_ancilla=clean_ancilla,
    )
    alias_res = synthesize_qrom(
        table.alias_indices, n, n,
        use_select_swap=use_select_swap,
        clean_ancilla=clean_ancilla,
    )

    nq = 0
    addr = list(range(nq, nq + n)); nq += n

    keep_start = nq
    nq += keep_res.qc.num_qubits
    keep_out = [keep_start + keep_res.output_qubits[i] for i in range(b)]

    rand_q = list(range(nq, nq + b)); nq += b
    flag = nq; nq += 1

    alias_start = nq
    nq += alias_res.qc.num_qubits
    alias_out = [alias_start + alias_res.output_qubits[i] for i in range(n)]

    qc = QuantumCircuit(nq)

    for i in range(n):
        qc.h(addr[i])
    for i in range(b):
        qc.h(rand_q[i])

    keep_map = list(range(keep_start, keep_start + keep_res.qc.num_qubits))
    for i in range(n):
        keep_map[keep_res.input_qubits[i]] = addr[i]
    qc.compose(keep_res.qc, qubits=keep_map, inplace=True)

    compare_ccx = add_comparator(qc, keep_out, rand_q, flag)

    alias_map = list(range(alias_start, alias_start + alias_res.qc.num_qubits))
    for i in range(n):
        alias_map[alias_res.input_qubits[i]] = addr[i]
    qc.compose(alias_res.qc, qubits=alias_map, inplace=True)

    cswap_ccx, cswap_cx = add_controlled_swap(qc, flag, addr, alias_out)

    t_per_ccx = 4 if clean_ancilla else 7
    elapsed = (time.monotonic() - start) * 1000.0

    result = AliasSamplingResult(
        qc=qc,
        t_count=(keep_res.t_count + alias_res.t_count +
                 compare_ccx * t_per_ccx + cswap_ccx * t_per_ccx),
        cnot_count=qc.count_ops().get("cx", 0),
        qubit_count=nq,
        keep_qrom_t=keep_res.t_count,
        alias_qrom_t=alias_res.t_count,
        compare_t=compare_ccx * t_per_ccx,
        cswap_t=cswap_ccx * t_per_ccx,
        compile_time_ms=elapsed,
    )
    return result


def estimate_alias_sampling_t_count(
    n_qubits: int,
    precision_bits: int,
    use_select_swap: bool = True,
    clean_ancilla: bool = True,
) -> int:
    N = 1 << n_qubits
    t_per_ccx = 4 if clean_ancilla else 7

    if use_select_swap:
        keep_lam = compute_optimal_lambda(N, precision_bits)
        keep_t = select_swap_qrom_t_count(N, precision_bits, keep_lam, t_per_ccx)
        alias_lam = compute_optimal_lambda(N, n_qubits)
        alias_t = select_swap_qrom_t_count(N, n_qubits, alias_lam, t_per_ccx)
    else:
        keep_t = select_qrom_t_count(N, t_per_ccx)
        alias_t = select_qrom_t_count(N, t_per_ccx)

    compare_ccx = precision_bits
    cswap_ccx = n_qubits

    return keep_t + alias_t + (compare_ccx + cswap_ccx) * t_per_ccx
