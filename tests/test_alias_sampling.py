"""Tests for alias sampling state preparation."""
import pytest

from qsp.state import w_state
from qsp.qrom.alias_sampling import (
    AliasTable,
    build_alias_table,
    estimate_alias_sampling_t_count,
    synthesize_alias_sampling,
)


class TestAliasTable:
    def test_uniform_distribution(self):
        state = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
        table = build_alias_table(state, 2, 8)
        assert table.n_bins == 4
        assert len(table.keep_values) == 4
        assert len(table.alias_indices) == 4

    def test_w_state_table(self):
        state, n = w_state(3)
        table = build_alias_table(state, n, 8)
        assert table.n_bins == 8
        assert table.precision_bits == 8


class TestAliasSampling:
    def test_synthesize_basic(self):
        state, n = w_state(3)
        res = synthesize_alias_sampling(state, n, precision_bits=4, use_select_swap=False)
        assert res.t_count > 0
        assert res.qubit_count > n

    def test_synthesize_selectswap(self):
        state, n = w_state(3)
        res = synthesize_alias_sampling(state, n, precision_bits=4, use_select_swap=True)
        assert res.t_count > 0
        assert res.qubit_count > n

    def test_estimate_basic(self):
        t = estimate_alias_sampling_t_count(3, 4, use_select_swap=False)
        assert t > 0

    def test_estimate_selectswap(self):
        t = estimate_alias_sampling_t_count(3, 4, use_select_swap=True)
        assert t > 0

    def test_selectswap_saves_t(self):
        t_basic = estimate_alias_sampling_t_count(4, 8, use_select_swap=False)
        t_swap = estimate_alias_sampling_t_count(4, 8, use_select_swap=True)
        assert t_swap <= t_basic
