# State Preparation Compilation

Quantum state preparation algorithms in Python/Qiskit: rotation-based (sparse & dense) and QROM-based (alias sampling with Select/SelectSwap).

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```python
from qsp.state import w_state
from qsp.rotation.sparse import prepare_state_sparse
from qsp.rotation.dense import prepare_state_dense
from qsp.qrom.alias_sampling import synthesize_alias_sampling

state, n = w_state(4)

# Rotation-based
qc_sparse = prepare_state_sparse(state, n)
qc_dense = prepare_state_dense(state, n)

# QROM-based (alias sampling)
result = synthesize_alias_sampling(state, n, precision_bits=8)
```

## Tests

```bash
pytest tests/ -v
```

## Benchmark

```bash
python -m experiments.compare_t_cost --n-max 5 -v
```

## Structure

- `src/qsp/state.py` — Sparse state representation and generators
- `src/qsp/rotation/` — Cardinality reduction (sparse) and qubit reduction (dense)
- `src/qsp/qrom/` — QROM synthesis (Select, SelectSwap) and alias sampling
- `experiments/` — T-cost comparison benchmark and plotting
