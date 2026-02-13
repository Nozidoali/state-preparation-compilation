"""Gate counting and verification utilities."""
from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qsp.state import State, to_statevector


def estimate_t_count(qc: QuantumCircuit) -> int:
    counts = qc.count_ops()
    t = counts.get("t", 0) + counts.get("tdg", 0)
    t += counts.get("ccx", 0) * 4
    return t


def count_cnots(qc: QuantumCircuit) -> int:
    return qc.count_ops().get("cx", 0)


def verify_state_preparation(
    qc: QuantumCircuit,
    target: State,
    n_bits: int,
    tol: float = 1e-6,
) -> float:
    sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
    target_vec = to_statevector(target, n_bits)

    sim_vec = np.real(sv.data[: 1 << n_bits])
    fidelity = float(np.abs(np.dot(sim_vec, target_vec)) ** 2)
    return fidelity
