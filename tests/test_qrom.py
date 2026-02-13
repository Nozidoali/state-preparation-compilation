"""Tests for QROM synthesis."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qsp.qrom.synthesis import (
    compute_optimal_lambda,
    select_qrom_t_count,
    select_swap_qrom_t_count,
    synthesize_basic_select,
    synthesize_qrom,
    synthesize_select_swap,
)


class TestTCountFormulas:
    def test_select_qrom_trivial(self):
        assert select_qrom_t_count(1) == 0

    def test_select_qrom_small(self):
        assert select_qrom_t_count(4, 4) == 2 * 3 * 4

    def test_select_swap_t_count(self):
        t = select_swap_qrom_t_count(8, 4, 2, 4)
        assert t > 0

    def test_optimal_lambda_trivial(self):
        assert compute_optimal_lambda(1, 4) == 1
        assert compute_optimal_lambda(2, 0) == 1

    def test_optimal_lambda_nontrivial(self):
        lam = compute_optimal_lambda(16, 8)
        assert lam >= 1
        assert (lam & (lam - 1)) == 0


class TestBasicSelect:
    def test_basic_select_4entries(self):
        data = [0b01, 0b10, 0b11, 0b00]
        res = synthesize_basic_select(data, 2, 2)
        assert res.qc.num_qubits == 2 + 4 + 2
        assert res.t_count > 0

    def test_basic_select_simulation(self):
        data = [0b01, 0b10, 0b11, 0b00]
        addr_bits = 2
        data_bits = 2
        res = synthesize_basic_select(data, addr_bits, data_bits)
        qc = res.qc

        for addr_val in range(4):
            prep_qc = type(qc)(qc.num_qubits)
            for bit in range(addr_bits):
                if (addr_val >> bit) & 1:
                    prep_qc.x(bit)
            prep_qc.compose(qc, inplace=True)

            sv = Statevector.from_label("0" * qc.num_qubits).evolve(prep_qc)
            probs = np.abs(sv.data) ** 2

            expected_val = data[addr_val]
            n = qc.num_qubits
            out_start = res.output_qubits[0]
            found = False
            for idx in range(len(probs)):
                if probs[idx] < 1e-6:
                    continue
                out_val = 0
                for b in range(data_bits):
                    if (idx >> res.output_qubits[b]) & 1:
                        out_val |= (1 << b)
                assert out_val == expected_val, (
                    f"addr={addr_val}: expected output {expected_val}, got {out_val}"
                )
                found = True
            assert found, f"addr={addr_val}: no nonzero probability states found"


class TestSelectSwap:
    def test_select_swap_4entries(self):
        data = [0b01, 0b10, 0b11, 0b00]
        res = synthesize_select_swap(data, 2, 2, 2)
        assert res.t_count > 0
        assert res.lambda_used == 2

    def test_select_swap_simulation(self):
        data = [0b01, 0b10, 0b11, 0b00]
        addr_bits = 2
        data_bits = 2
        lam = 2
        res = synthesize_select_swap(data, addr_bits, data_bits, lam)
        qc = res.qc

        for addr_val in range(4):
            prep_qc = type(qc)(qc.num_qubits)
            for bit in range(addr_bits):
                if (addr_val >> bit) & 1:
                    prep_qc.x(bit)
            prep_qc.compose(qc, inplace=True)

            sv = Statevector.from_label("0" * qc.num_qubits).evolve(prep_qc)
            probs = np.abs(sv.data) ** 2

            expected_val = data[addr_val]
            found = False
            for idx in range(len(probs)):
                if probs[idx] < 1e-6:
                    continue
                out_val = 0
                for b in range(data_bits):
                    if (idx >> res.output_qubits[b]) & 1:
                        out_val |= (1 << b)
                assert out_val == expected_val, (
                    f"addr={addr_val}: expected output {expected_val}, got {out_val}"
                )
                found = True
            assert found, f"addr={addr_val}: no nonzero probability states found"

    def test_synthesize_qrom_auto(self):
        data = [1, 2, 3, 4, 5, 6, 7, 0]
        res = synthesize_qrom(data, 3, 3)
        assert res.t_count > 0
