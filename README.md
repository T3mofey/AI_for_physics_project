# Generative modeling of correlated financial returns: from classical RBMs to a variational quantum Born machine

Individual project for **AI Models for Physics (2025/26)**, Università degli Studi di Milano
(Prof. Enrico Prati, Dr. Sebastiano Corli).

## Overview

Can a generative model learn the *joint* distribution of the daily up/down moves of a basket
of correlated stocks? This project answers that with two models trained on the same data and
compared on the same metric:

1. a **Restricted Boltzmann Machine (RBM)** implemented from scratch in PyTorch and trained with
   Contrastive Divergence, first **validated on the 2D Ising model** so we know the implementation
   reproduces known physics before we trust it on finance; and
2. a small **variational quantum circuit trained as a Born machine** (PennyLane), used as a
   quantum-generative baseline on the identical task.

Each day's vector of asset signs is treated as a spin configuration, making the financial problem
directly analogous to the Ising problem. The headline comparison is how faithfully each model
reproduces the **pairwise correlation matrix** of the real returns.

Course topics touched: 2D Ising model (finite *T*), the Metropolis algorithm, Gibbs sampling,
Restricted Boltzmann Machines and their training, and quantum machine learning for econophysics.

## Results

| Phase | Model | Task | Metric | Result |
|------|-------|------|--------|--------|
| 1 | RBM (CD-10, ±1 units) | reproduce 2D Ising thermodynamics at *T* ≈ *T*c | magnetization \|m\| | **0.686** (data 0.675) |
| 1 | RBM | same | energy / spin | **−1.274** (data −1.434) |
| 2 | RBM (CD-1, binary) | reproduce stock correlation matrix | correlation MAE | **0.0335** |
| 3 | Quantum Born machine | reproduce stock correlation matrix | correlation MAE (exact) | **0.0019 ± 0.0008** (5 seeds) |
| 3 | Quantum Born machine | same | correlation MAE (5000 shots) | **0.0109 ± 0.0021** |

**Reading the numbers.** Phase 1 validates the RBM: trained on Metropolis samples at the critical
temperature, its generated configurations recover the magnetization almost exactly and the energy
to within ~13% — the residual energy gap is expected, since *T*c is the hardest point for a finite
RBM (critical correlations are long-range). With the implementation validated, Phase 2 applies the
same RBM to binarized returns and reproduces the asset correlation matrix to a mean absolute error
of 0.033.

Phase 3 reports two MAEs for the Born machine because they answer different questions. The
**exact** MAE is computed directly from the circuit's full probability vector (possible because
this is a 6-qubit simulator) and reflects the model's true correlation error. The **sampled** MAE
is estimated from 5000 measurement shots and is the fair, apples-to-apples comparison against the
RBM, which can only be evaluated by sampling; it sits higher than the exact value purely because of
finite-shot noise. On the like-for-like sampled comparison the Born machine (0.011) and the RBM
(0.034) are in the same ballpark.

**Takeaway.** At this scale (6 assets / 6 qubits, classical simulation) a small variational quantum
Born machine *matches* a classical RBM at reproducing the pairwise correlation structure of
binarized returns. This is a parity result and is **not** evidence of quantum advantage — at six
qubits on a simulator no such claim would be warranted.

Figures produced by the notebook live in [`figures/`](figures/):

- `ising_thermodynamics.png` — magnetization and energy vs temperature (Ising validation).
- `correlation_comparison.png` — real vs RBM-generated correlation matrices.
- `born_machine_loss.png` — Born machine MMD² training curve.
- `quantum_correlation.png` — real vs Born-machine correlation matrices.

## Repository layout

```
.
├── notebooks/
│   └── 00_combined.ipynb   # single self-contained notebook (all code inlined)
├── figures/
└── requirements.txt
```

## Method

**Ising sampler.** Single-spin-flip Metropolis Monte Carlo on an *L×L* lattice with periodic
boundary conditions. Generates training configurations across a temperature sweep around *T*c ≈ 2.269
and computes magnetization and energy-per-spin observables.

**RBM.** Energy *E*(v, h) = −aᵀv − bᵀh − vᵀW h, with conditional distributions
*p*(hⱼ=1|v) = σ(bⱼ + Wⱼᵀv) and *p*(vᵢ=1|h) = σ(aᵢ + Wᵢh). Trained by Contrastive Divergence: a
*k*-step block-Gibbs chain produces the negative-phase statistics. Supports ±1 (Ising) and {0,1}
(binary) units.

**Financial data.** Adjusted close prices for AAPL, MSFT, GOOGL, AMZN, META, NVDA
(2018-01-01 to 2024-01-01) via `yfinance`; daily log-returns binarized to ±1 by sign (1508 trading
days, fraction-up ≈ 0.533). The RBM is **generative** — it learns *P*(asset₁, …, assetₙ), not a
return prediction.

**Quantum Born machine.** Hardware-efficient ansatz — per-layer RY/RZ rotations with a **circular**
CNOT entangling block, plus a final rotation layer — measured in the computational basis. Trained to
minimize the **Maximum Mean Discrepancy (MMD)** against the empirical data distribution, with
RBF-kernel bandwidths set by the median heuristic. Because the simulator exposes the full state,
correlations can be read exactly from the probability vector in addition to being estimated from
samples.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/T3mofey/AI_for_physics_project.git
cd AI_for_physics_project
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Running

```bash
jupyter notebook        # then open notebooks/00_combined.ipynb
```

All four phases run top-to-bottom in a single notebook. The multi-seed Born machine cell takes a
few minutes on CPU. Random seeds are fixed, so results are reproducible; small differences from the
table above can arise from the live `yfinance` download.

## Notes and limitations

- Results are at small scale by design (6 assets / 6 qubits). The quantum/classical comparison is a
  *parity* finding on a classical simulator, not a hardware result and not a claim of advantage.
- The RBM validation is reported at *T*c, the most demanding temperature; agreement away from the
  critical point is tighter.
- The MMD objective and the median-heuristic kernel follow the differentiable-Born-machine recipe
  of Liu & Wang (2018).

## References

1. Hinton, *A Practical Guide to Training RBMs* (2010).
2. Torlai & Melko, *Learning thermodynamics with Boltzmann machines*, PRB 94, 165134 (2016). arXiv:1606.02718
3. Carrasquilla & Melko, *Machine learning phases of matter*, Nature Physics 13, 431 (2017). arXiv:1605.01735
4. Liu & Wang, *Differentiable Learning of Quantum Circuit Born Machines*, PRA 98, 062324 (2018). arXiv:1804.04168
5. Coyle et al., *The Born Supremacy: Quantum Advantage and Training of an Ising Born Machine* (2019). arXiv:1904.02214
6. Amin et al., *Quantum Boltzmann Machine*, PRX 8, 021050 (2018). arXiv:1601.02036
