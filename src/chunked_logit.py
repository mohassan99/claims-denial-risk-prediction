"""
Memory-bounded Newton-Raphson logistic regression -- ordinary MLE or Firth's
penalized MLE -- with support for fitting a model whose coefficients on a
chosen subset of columns are fixed at 0 (a nested/restricted fit that keeps
the FULL model's Firth penalty).

WHY THIS EXISTS (2026-09-24). Three problems surfaced in the first smoke
tests on the full-rank Chow matrices, none fixable by a flag:

  1. statsmodels' lbfgs does not actually fit these matrices. Dollar-amount
     columns reach ~$410,000 next to 0/1 dummies; lbfgs overflowed exp() on
     its first steps and then reported converged=True at a log-likelihood
     WORSE than the intercept-only model (e.g. -15,969 at --sample-frac
     0.02, where predicting the 12.1% base rate for everyone gives about
     -8,516). The same symptom -- identical restricted and unrestricted
     log-likelihoods, LR = 0 -- is in every earlier standard-method run.
     Newton-Raphson on column-scaled data is immune: the MLE is invariant
     to column scaling, and Newton converges quadratically.
  2. firthmodels cannot fit the full-size unrestricted matrix on an 8 GB
     machine: its workspace allocates three extra k x n float64 buffers on
     top of X itself (~10 GB at 1.15M rows x 271 columns). Every quantity
     Firth needs -- X'WX, the hat diagonals h_i, the modified score -- is a
     sum over rows, so this module accumulates them over row chunks.
  3. Two separately-penalized Firth fits do not give Firth's penalized
     likelihood-ratio test. The penalty 0.5*log|X'WX| depends on the
     dimension and parameterization of each model's own design, so the
     difference of two separately-penalized log-likelihoods mixes a
     penalty difference into the LR statistic (at --sample-frac 0.02: 339.1
     that way vs. 213.2 for the proper test). The standard penalized LR
     test (Heinze & Schemper 2002; what logistf and firthmodels' own
     per-coefficient .lrt() do) fits the restricted model as the full model
     with the tested coefficients held at 0, penalized by the FULL model's
     information matrix. fit(..., fixed_zero=mask) does exactly that; the
     Chow test is put in that form by reparameterizing the unrestricted
     design (see fit_chow_test.py, _build_reparameterized_design).

VALIDATED against firthmodels 0.8.2 on the --sample-frac 0.02 matrices
before any real run (see FEATURE_ENGINEERING.md Section 6, 2026-09-24):
penalized and unpenalized log-likelihoods agree with firthmodels' fits,
and the fixed-zero fit agrees with a constrained fit built from
firthmodels' own compute_logistic_quantities + newton_raphson.

Numerics. Columns are divided by their max |value| (held in self.scale)
for conditioning; X is modified IN PLACE (the caller hands over its
array -- no second copy on an 8 GB machine). Reported log-likelihoods are
in the ORIGINAL units: ordinary log-likelihood is scale-invariant, and the
Firth penalty's scale-dependence (log|D X'WX D| = log|X'WX| + 2 sum log d)
is added back. Step control mirrors firthmodels: Fisher-scoring step
(= Newton for the logit link), clipped to max_step per coefficient,
step-halving until the (penalized) log-likelihood does not decrease;
converged when max|score| < gtol AND max|step| < xtol. max_step defaults
to 20 (firthmodels uses 5): on the scaled columns, 5 made fits take ~100
clipped iterations to walk large coefficients out, while 20 reached the
identical optimum (penalized log-likelihood equal to ~1e-10 at 0.02) in
roughly a third of the iterations. Step-halving, not the clip, is what
guarantees ascent. The full-data Stage 1 Firth fits were run at 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import expit


@dataclass
class FitResult:
    llf: float                # (penalized, if Firth) log-likelihood, original units
    llf_unpenalized: float    # ordinary log-likelihood at the same coefficients
    beta: np.ndarray          # coefficients in ORIGINAL column units
    converged: bool
    n_iter: int
    max_abs_score: float
    newton_decrement: float   # score' I^-1 score over the free coefficients
    nobs: int
    mle_retvals: dict = field(default_factory=dict)


class ChunkedLogit:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        penalty_weight: float = 0.5,
        chunk_rows: int = 50_000,
    ) -> None:
        if X.dtype != np.float64 or not X.flags.c_contiguous:
            raise ValueError("X must be a C-contiguous float64 array (it is scaled in place).")
        self.X = X
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.n, self.k = X.shape
        self.pw = float(penalty_weight)
        self.chunk_rows = chunk_rows

        scale = np.zeros(self.k)
        for s in range(0, self.n, chunk_rows):
            np.maximum(scale, np.abs(X[s : s + chunk_rows]).max(axis=0), out=scale)
        if (scale == 0).any():
            raise ValueError("All-zero column(s) in X -- the design is not full rank.")
        for s in range(0, self.n, chunk_rows):
            X[s : s + chunk_rows] /= scale
        self.scale = scale
        # log|X'WX| in original units = log|scaled| + 2 * sum(log scale)
        self._logdet_offset = 2.0 * float(np.log(scale).sum())

    def _quantities(self, beta: np.ndarray):
        """(penalized ll, unpenalized ll, modified score, Fisher info) at
        `beta` (scaled units, all k coefficients). Two chunked passes when
        penalized: X'WX first, then the hat diagonals that need its inverse."""
        n, k, X, y = self.n, self.k, self.X, self.y
        eta = np.empty(n)
        for s in range(0, n, self.chunk_rows):
            np.dot(X[s : s + self.chunk_rows], beta, out=eta[s : s + self.chunk_rows])
        p = expit(eta)
        w = p * (1.0 - p)
        ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        info = np.zeros((k, k))
        for s in range(0, n, self.chunk_rows):
            Xc = X[s : s + self.chunk_rows]
            info += (Xc * w[s : s + self.chunk_rows, None]).T @ Xc
        resid = y - p
        if self.pw == 0.0:
            score = np.zeros(k)
            for s in range(0, n, self.chunk_rows):
                score += X[s : s + self.chunk_rows].T @ resid[s : s + self.chunk_rows]
            return ll, ll, score, info

        try:
            L = np.linalg.cholesky(info)
        except np.linalg.LinAlgError:
            # A trial step so large the weights underflowed and X'WX lost
            # positive-definiteness: the penalty log|X'WX| is -inf there, so
            # the penalized likelihood is -inf -- report that, and fit()'s
            # step-halving backs off. (Found 2026-09-24 testing an unclipped
            # step on the 0.02 data.)
            return -np.inf, ll, np.zeros(k), info
        logdet = 2.0 * float(np.log(np.diag(L)).sum())
        score = np.zeros(k)
        for s in range(0, n, self.chunk_rows):
            Xc = X[s : s + self.chunk_rows]
            V = solve_triangular(L, Xc.T, lower=True, check_finite=False)
            h = w[s : s + self.chunk_rows] * np.einsum("ij,ij->j", V, V)
            del V
            r = resid[s : s + self.chunk_rows] + self.pw * h * (1.0 - 2.0 * p[s : s + self.chunk_rows])
            score += Xc.T @ r
        ll_pen = ll + self.pw * (logdet + self._logdet_offset)
        return ll_pen, ll, score, info

    def fit(
        self,
        fixed_zero: np.ndarray | None = None,
        beta_init: np.ndarray | None = None,
        max_iter: int = 200,
        max_step: float = 20.0,
        max_halfstep: int = 25,
        gtol: float = 1e-4,
        xtol: float = 1e-4,
        verbose: bool = True,
    ) -> FitResult:
        """Fit, with coefficients where fixed_zero is True held at 0 (the
        penalty still uses all k columns). beta_init is in ORIGINAL units."""
        free = np.ones(self.k, dtype=bool) if fixed_zero is None else ~np.asarray(fixed_zero, dtype=bool)
        fidx = np.flatnonzero(free)
        beta = np.zeros(self.k)
        if beta_init is not None:
            beta = np.asarray(beta_init, dtype=np.float64) * self.scale
            beta[~free] = 0.0

        ll, llu, score, info = self._quantities(beta)
        converged = False
        it = 0
        for it in range(max_iter + 1):
            g = score[fidx]
            try:
                c = cho_factor(info[np.ix_(fidx, fidx)])
            except np.linalg.LinAlgError:
                # Fisher information lost positive-definiteness: weights
                # collapsing to 0 as coefficients run off to infinity, i.e.
                # (quasi-)separation -- the ordinary MLE does not exist.
                if verbose:
                    print("      Fisher information singular (separation?) -- stopping, NOT converged")
                break
            delta = cho_solve(c, g)
            max_g = float(np.abs(g).max())
            max_d = float(np.abs(delta).max())
            decrement = float(g @ delta)
            if verbose:
                print(f"      iter {it:3d}: ll {ll:.6f}  max|score| {max_g:.3g}  max|step| {max_d:.3g}", flush=True)
            if max_g < gtol and max_d < xtol:
                converged = True
                break
            if it == max_iter:
                break
            if max_d > max_step:
                delta *= max_step / max_d
            step = 1.0
            for _ in range(max_halfstep + 1):
                cand = beta.copy()
                cand[fidx] += step * delta
                q = self._quantities(cand)
                if q[0] >= ll:
                    break
                step *= 0.5
            else:
                if verbose:
                    print("      step-halving failed -- stopping, NOT converged")
                break
            beta = cand
            ll, llu, score, info = q

        g = score[fidx]
        try:
            decrement = float(g @ cho_solve(cho_factor(info[np.ix_(fidx, fidx)]), g))
        except np.linalg.LinAlgError:
            decrement = float("nan")
        return FitResult(
            llf=ll,
            llf_unpenalized=llu,
            beta=beta / self.scale,
            converged=converged,
            n_iter=it,
            max_abs_score=float(np.abs(g).max()),
            newton_decrement=decrement,
            nobs=self.n,
            mle_retvals={"converged": converged, "iterations": it},
        )
