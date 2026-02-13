"""Shared circuit primitives: comparator and controlled-swap."""
from __future__ import annotations

from qiskit.circuit import QuantumCircuit


def add_comparator(
    qc: QuantumCircuit,
    a_qubits: list[int],
    b_qubits: list[int],
    flag_qubit: int,
) -> int:
    bits = len(a_qubits)
    assert len(a_qubits) == len(b_qubits)
    ccx_count = 0

    qc.x(flag_qubit)
    for i in range(bits - 1, -1, -1):
        qc.ccx(a_qubits[i], flag_qubit, b_qubits[i])
        ccx_count += 1
        qc.cx(a_qubits[i], flag_qubit)

    return ccx_count


def add_controlled_swap(
    qc: QuantumCircuit,
    ctrl: int,
    reg_a: list[int],
    reg_b: list[int],
) -> tuple[int, int]:
    assert len(reg_a) == len(reg_b)
    ccx_count = 0
    cx_count = 0
    for i in range(len(reg_a)):
        qc.cx(reg_a[i], reg_b[i])
        qc.ccx(ctrl, reg_b[i], reg_a[i])
        qc.cx(reg_a[i], reg_b[i])
        ccx_count += 1
        cx_count += 2
    return ccx_count, cx_count
