# Asymptotic analysis of SAA for optimal control with uncertain dynamics: numerical experiments

This repository contains the code and the data that reproduce the numerical results of the paper

> M. Soledad Aronna, João Miguel Machado and Oscar Sierra,
> *Asymptotic analysis of sample average approximations for optimal control problems with uncertain dynamics*,
> arXiv:XXXX.XXXXX (2026).

The paper studies the sample average approximation (SAA) of optimal control problems in which a single
open-loop control steers a family of control-affine systems with uncertain parameters, with a Tikhonov
term $\frac{\alpha}{2}\|u\|_{L^2}^2$ and pointwise control constraints. It proves central limit theorems for
the optimal values and for the optimal controls. The experiments compare the Monte Carlo behaviour of the
SAA estimators with these asymptotic predictions.

## Examples

| Example | Model | Uncertain parameters | Regimes |
|---|---|---|---|
| `oscillator` | $\dot x_1=x_2$, $\dot x_2=-\omega^2x_1+u$, $x(0)=(1,0)$, cost $\tfrac12\mathbb E\lvert x(T)\rvert^2$, $\lvert u\rvert\le 1$, $T=3$ | $\omega\sim\mathcal U[0.5,1.5]$ | $\alpha=0.3$ (inactive constraints), $\alpha=0.05$ (active constraints) |
| `sit` | sterile insect technique model with the parameters of Bidi, Coron, Hayat and Lichtle (Table 1); cost $\mathbb E[F(T)+M(T)]$, $0\le u\le 1$, $T=90$ days | competitiveness $q\sim\mathcal U[0.91,1]$; sterile-male death rate $\delta_S\sim\mathcal N(0.12,0.01^2)$ truncated to $[0.10,0.14]$ | $\alpha=0.1$ (inactive), $\alpha=0.03$ (active upper bound) |

For each example and each value of $\alpha$, the script

1. solves the "true" problem with the expectation computed by tensor Gauss–Legendre quadrature
   (200 nodes for the oscillator, $20\times 20$ nodes for the SIT example);
2. computes the asymptotic quantities of the central limit theorems: the variance $\sigma^2(\hat u)$
   of the optimal value and the covariance $\Sigma(\hat u)$ of the optimal control;
3. solves the SAA problem for sample sizes $M=10,20,\dots$ over many independent replications
   (1000 for the oscillator, 500 for the SIT example) and records the errors, the normalized
   fluctuations $\sqrt M\langle\hat u_M-\hat u,h_i\rangle$ for $h_1\equiv1$ and $h_2(t)=\cos(\pi t/T)$,
   and the coverage of the 95% confidence intervals given by the plug-in covariance estimator.

The controls are piecewise constant on a uniform grid. For the oscillator the dynamics is integrated
exactly and each SAA problem is a box-constrained quadratic program. For the SIT example the dynamics is
integrated by a semi-implicit Euler scheme, and the gradients are computed exactly for the discrete scheme
by the discrete adjoint equation.

## Requirements

Python 3.9 or later with NumPy, SciPy and Matplotlib:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python saa_numerics.py                          # redraw both figures from results/*.json (a few seconds)
python saa_numerics.py --run oscillator         # rerun the oscillator with the settings of the paper
python saa_numerics.py --run sit                # rerun the SIT example with the settings of the paper
python saa_numerics.py --run sit --parallel     # same, one process per value of alpha (Linux/macOS)
python saa_numerics.py --run all --mode quick   # fast test run with fewer replications
python saa_numerics.py --help
```

Measured running times with the settings of the paper (Linux, one core per process): about 3 minutes for
the oscillator, and about 30 minutes for the SIT example (about 22 minutes with `--parallel` on two cores);
`--mode quick` takes under 2 minutes for each example. The SIT run writes its progress to
`results/progress_alpha_<alpha>.txt`.

## Repository contents

| File | Content |
|---|---|
| `saa_numerics.py` | all the code (models, solvers, experiments and figures) |
| `results/oscillator.json` | results of the oscillator experiment reported in the paper |
| `results/sit_paper.json` | results of the SIT experiment reported in the paper |
| `figures/fig_oscillator.pdf`, `figures/fig_sit_paper.pdf` | figures of the paper, drawn from the files above |

Main fields of the `.json` files: `ubar` (optimal control $\hat u$), `V` (optimal value), `sig2V`
($\sigma^2(\hat u)$), `trSig` ($\operatorname{tr}\Sigma(\hat u)$), `pred` (predicted variances
$\langle\Sigma(\hat u)h_i,h_i\rangle$), `pstd` (predicted pointwise standard deviation), `M` (sample
sizes), `EabsdV` ($\mathbb E\lvert V_M-V\rvert$), `Eerr2` ($\mathbb E\|\hat u_M-\hat u\|^2$), `zvar`
(empirical variances of $\sqrt M\langle\hat u_M-\hat u,h_i\rangle$), `devstd` (empirical pointwise standard
deviation of $\sqrt M(\hat u_M-\hat u)$) and `coverage` (empirical coverage of the 95% intervals).

## Reproducibility

All random numbers are generated from fixed seeds (`numpy.random.default_rng`), so a rerun draws the same
samples. A complete rerun of both experiments on a different platform (Python 3.13, NumPy 2.5, SciPy 1.18)
reproduced the files in `results/` up to solver tolerances: relative differences below $10^{-8}$ in all
reported statistics, and identical coverage counts. Each new run stores the library versions and the
running time in the field `meta` of the output file.

## How to cite

If you use this code, please cite the paper above and this software (see `CITATION.cff`; the archived
version has the DOI 10.5281/zenodo.XXXXXXX).

## License

The code is distributed under the MIT License (see `LICENSE`).
