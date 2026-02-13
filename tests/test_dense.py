"""Tests for dense (qubit reduction) state preparation."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qsp.state import dense_uniform, dicke_state, to_statevector, w_state
from qsp.rotation.dense import prepare_state_dense


def _fidelity(qc, target, n_bits):
    sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
    sim = np.real(sv.data[: 1 << n_bits])
    tgt = to_statevector(target, n_bits)
    return float(np.abs(np.dot(sim, tgt)) ** 2)


class TestDense:
    def test_w_state_3(self):
        state, n = w_state(3)
        qc = prepare_state_dense(state, n, timeout=30.0)
        assert qc.num_qubits == n
        fid = _fidelity(qc, state, n)
        assert fid > 1 - 1e-6, f"fidelity={fid}"

    def test_w_state_4(self):
        state, n = w_state(4)
        qc = prepare_state_dense(state, n, timeout=30.0)
        assert qc.num_qubits == n
        fid = _fidelity(qc, state, n)
        assert fid > 1 - 1e-6, f"fidelity={fid}"

    def test_dicke_3_2(self):
        state, n = dicke_state(3, 2)
        qc = prepare_state_dense(state, n, timeout=30.0)
        assert qc.num_qubits == n
        fid = _fidelity(qc, state, n)
        assert fid > 1 - 1e-6, f"fidelity={fid}"

    def test_dense_uniform_3(self):
        state, n = dense_uniform(3)
        qc = prepare_state_dense(state, n, timeout=30.0)
        assert qc.num_qubits == n
        fid = _fidelity(qc, state, n)
        assert fid > 1 - 1e-6, f"fidelity={fid}"
