# Distributed Lasso Support Recovery

This project studies privacy-preserving sparse regression across multiple data owners. Each node keeps its feature matrix and response vector local, selects a local Lasso regularization parameter, and runs coordinate-descent updates initialized from a shared global model. A central aggregator receives only coefficient vectors and scalar diagnostics, then forms a sample-size-weighted consensus estimate.

The repository is a sanitized and reproducible portfolio version of a statistical computing project. The original course data is intentionally not included. A synthetic-data example is provided so the full workflow can be run without restricted or private data.

## Problem

Suppose three organizations want to estimate one sparse regression vector without exchanging raw observations. At communication round `t`:

1. The server broadcasts the current consensus vector.
2. Each node runs `E` local coordinate-descent epochs using its private data and fixed regularization parameter.
3. Nodes return updated coefficient vectors.
4. The server computes a weighted average based on local sample sizes.
5. Training stops when the Euclidean change in the global vector falls below a tolerance.

The main experiment compares `E = 1, 5, 10`, illustrating the tradeoff between communication efficiency and node drift.

## Original Experiment Results

The original three-node experiment used 600 predictors and 800 pooled training observations. The data itself is not distributed here.

| Local epochs `E` | Communication rounds | Final weighted loss | Active coefficients | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27 | 20.3103 | 401 | 156 | 245 | 24 | 175 |
| 5 | 9 | 18.9366 | 324 | 138 | 186 | 42 | 234 |
| 10 | 7 | 18.8788 | 311 | 130 | 181 | 50 | 239 |

More local work reduced the number of communication rounds, but it also moved the distributed solution away from the pooled reference support. `E = 1` recovered more reference-active coefficients, while `E = 5` provided a practical compromise between communication cost and support recovery.

## Repository Structure

```text
.
├── data/README.md                 # Data contract; no original data
├── examples/run_synthetic_demo.py # End-to-end reproducible example
├── figures/                       # Original aggregate result plots
├── src/distributed_lasso.py       # Optimization and evaluation logic
├── tests/test_distributed_lasso.py
├── requirements.txt
└── README.md
```

## Run the Synthetic Demo

```bash
python -m pip install -r requirements.txt
python examples/run_synthetic_demo.py
```

The demo generates three heterogeneous nodes, tunes local regularization parameters, trains distributed models for several values of `E`, fits a pooled Lasso reference, and prints support-recovery metrics.

## Key Implementation Details

- Lasso objective: `(1 / 2m) * ||y - X beta||^2 + lambda * ||beta||_1`
- Sequential coordinate descent with residual updates
- Local validation-based regularization selection
- Sample-size-weighted synchronous aggregation
- Tolerance-based convergence
- Support comparison against a pooled reference estimator

## Data and Privacy

- No original course dataset is included.
- No personal, customer, or institutional records are included.
- No credentials, API keys, student identifiers, or absolute local paths are included.
- `data/README.md` documents the expected format for users who have their own authorized data.

Before applying this code to real distributed data, add secure transport, authentication, access controls, and formal privacy mechanisms. Sharing model parameters alone does not guarantee differential privacy.

