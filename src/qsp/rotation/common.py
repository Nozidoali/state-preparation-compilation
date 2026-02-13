"""Shared helpers for rotation-based state preparation.

Includes pivot selection, don't-care optimization, and sparse state simulation
functions that manipulate ``dict[int, float]`` to track the evolving state
during iterative reduction.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from qsp.state import EPS, State


# --- State simulation on dict[int, float] ---

def apply_x(state: State, qubit: int) -> State:
    return {idx ^ (1 << qubit): amp for idx, amp in state.items()}


def apply_cx(state: State, control: int, target: int) -> State:
    new: State = {}
    for idx, amp in state.items():
        if (idx >> control) & 1:
            new[idx ^ (1 << target)] = amp
        else:
            new[idx] = amp
    return new


def apply_controlled_cx(state: State, control: int, ctrl_val: bool, target: int) -> State:
    new: State = {}
    for idx, amp in state.items():
        bit = (idx >> control) & 1
        if bool(bit) == ctrl_val:
            new[idx ^ (1 << target)] = amp
        else:
            new[idx] = amp
    return new


def apply_ry(state: State, qubit: int, theta: float) -> State:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    new: State = {}
    for idx, amp in state.items():
        idx0 = idx & ~(1 << qubit)
        idx1 = idx | (1 << qubit)
        if (idx >> qubit) & 1:
            new[idx0] = new.get(idx0, 0.0) - s * amp
            new[idx1] = new.get(idx1, 0.0) + c * amp
        else:
            new[idx0] = new.get(idx0, 0.0) + c * amp
            new[idx1] = new.get(idx1, 0.0) + s * amp
    return _prune(new)


def apply_mcry(state: State, ctrls: List[int], phases: List[bool],
               theta: float, target: int) -> State:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    new: State = {}
    for idx, amp in state.items():
        match = all(bool((idx >> q) & 1) == p for q, p in zip(ctrls, phases))
        if match:
            idx0 = idx & ~(1 << target)
            idx1 = idx | (1 << target)
            if (idx >> target) & 1:
                new[idx0] = new.get(idx0, 0.0) - s * amp
                new[idx1] = new.get(idx1, 0.0) + c * amp
            else:
                new[idx0] = new.get(idx0, 0.0) + c * amp
                new[idx1] = new.get(idx1, 0.0) + s * amp
        else:
            new[idx] = new.get(idx, 0.0) + amp
    return _prune(new)


def apply_ucry(state: State, control_indices: List[int],
               angles: List[float], target: int) -> State:
    new: State = {}
    for idx, amp in state.items():
        rot_idx = 0
        for i, q in enumerate(control_indices):
            if (idx >> q) & 1:
                rot_idx |= 1 << i
        theta = angles[rot_idx]
        c = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        idx0 = idx & ~(1 << target)
        idx1 = idx | (1 << target)
        if (idx >> target) & 1:
            new[idx0] = new.get(idx0, 0.0) - s * amp
            new[idx1] = new.get(idx1, 0.0) + c * amp
        else:
            new[idx0] = new.get(idx0, 0.0) + c * amp
            new[idx1] = new.get(idx1, 0.0) + s * amp
    return _prune(new)


def _prune(state: State) -> State:
    return {k: v for k, v in state.items() if abs(v) > EPS}


# --- Pivot selection helpers ---

def maximize_difference_once(
    n_bits: int,
    indices: Set[int],
    diff_values: Dict[int, bool],
) -> Tuple[int, bool]:
    max_diff = -1
    max_diff_indices_1: Set[int] = set()
    max_diff_qubit = 0
    max_diff_value = False
    length = len(indices)

    for qubit in range(n_bits):
        if qubit in diff_values:
            continue
        indices_1 = {idx for idx in indices if (idx >> qubit) & 1}
        diff = abs(length - 2 * len(indices_1))
        if diff == length:
            continue
        if diff > max_diff:
            max_diff = diff
            max_diff_indices_1 = indices_1
            max_diff_qubit = qubit
            max_diff_value = length > 2 * len(indices_1)
        if max_diff == length - 1:
            break

    if max_diff_value:
        indices.clear()
        indices.update(max_diff_indices_1)
    else:
        indices -= max_diff_indices_1

    diff_values[max_diff_qubit] = max_diff_value
    return max_diff_qubit, max_diff_value


def select_informative_qubit(state: State, n_bits: int, supports: List[int]) -> int:
    best_qubit = 0
    min_diff = float("inf")
    length = len(state)
    assert length >= 2

    for qubit in supports:
        count0 = sum(1 for idx in state if not ((idx >> qubit) & 1))
        diff = abs(length - 2 * count0)
        if diff < min_diff:
            min_diff = diff
            best_qubit = qubit

    return best_qubit


def rotation_angles_optimization(
    angles: List[float],
    control_indices: List[int],
) -> Tuple[List[float], List[int]]:
    n_ctrls = len(control_indices)
    assert len(angles) == (1 << n_ctrls)

    min_sep = 1e-6
    dont_cares: Set[int] = set()

    for i in range(n_ctrls):
        is_dc = True
        for rot_idx in range(len(angles)):
            rev = rot_idx ^ (1 << i)
            if abs(angles[rev] - angles[rot_idx]) > min_sep:
                is_dc = False
                break
        if is_dc:
            dont_cares.add(i)

    if not dont_cares:
        return angles, control_indices

    new_ctrls: List[int] = []
    old_positions: List[int] = []
    for old_i in range(n_ctrls):
        if old_i not in dont_cares:
            new_ctrls.append(control_indices[old_i])
            old_positions.append(old_i)

    new_angles: List[float] = []
    for new_idx in range(1 << len(new_ctrls)):
        old_idx = 0
        for i, old_pos in enumerate(old_positions):
            old_idx |= ((new_idx >> i) & 1) << old_pos
        new_angles.append(angles[old_idx])

    return new_angles, new_ctrls


# --- Qubit-signature helpers for dense strategy ---

def get_qubit_signatures(state: State, n_bits: int) -> List[int]:
    sigs = [0] * n_bits
    for idx in sorted(state.keys()):
        for j in range(n_bits):
            sigs[j] = (sigs[j] << 1) | ((idx >> j) & 1)
    return sigs


def get_const1_signature(state: State) -> int:
    return (1 << len(state)) - 1


def get_ap_ry_angle(state: State, n_bits: int, qubit: int) -> Optional[float]:
    theta: Optional[float] = None
    for idx in state:
        rev = idx ^ (1 << qubit)
        if rev not in state:
            return None
        idx0 = idx & ~(1 << qubit)
        idx1 = idx | (1 << qubit)
        if idx0 not in state or idx1 not in state:
            return None
        w0 = state[idx0]
        w1 = state[idx1]
        t = 2.0 * math.atan(w1 / w0) if abs(w0) > EPS else math.pi
        if theta is None:
            theta = t
        elif abs(theta - t) > 1e-10:
            return None
    return theta
