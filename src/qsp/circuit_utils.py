"""Shared circuit primitives: comparator and controlled-swap."""
from __future__ import annotations

from qiskit.circuit import QuantumCircuit


def add_comparator(
    qc: QuantumCircuit,
    a_qubits: list[int],
    b_qubits: list[int],
    flag_qubit: int,
) -> int:
    """Compare a < b using carry chain. Sets flag=1 iff a < b.

    Uses Cuccaro-style MAJ gates to propagate the borrow of (a - b).
    flag must start at |0>. Modifies b_qubits in place (treated as scratch).
    """
    bits = len(a_qubits)
    assert len(a_qubits) == len(b_qubits)
    ccx_count = 0

    for i in range(bits):
        qc.x(a_qubits[i])

    for i in range(bits):
        qc.cx(flag_qubit, b_qubits[i])
        qc.cx(flag_qubit, a_qubits[i])
        qc.ccx(a_qubits[i], b_qubits[i], flag_qubit)
        ccx_count += 1

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
