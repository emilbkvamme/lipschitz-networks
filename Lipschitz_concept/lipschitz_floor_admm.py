import warnings
import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

##### LIPSCHITZ FLOOR (1D) #####
# The "floor" is the lowest MSE that ANY L-Lipschitz function can achieve on the data points (x_i, y_i):
#
#     minimise  sum_i (g_i - y_i)^2    subject to    |g_{i+1} - g_i| <= L * (x_{i+1} - x_i)
#
# In 1D it is enough to constrain neighbouring points (after sorting); all other pairs follow from the
# triangle inequality. The problem is convex, so it has one global optimum.
#
# How it is solved:
#   1) ADMM: split the problem into "fit the data" and "keep the slopes within +-L", and alternate between them.
#      The first step is a tridiagonal linear solve, the second is clipping the slopes. Both cost O(n).
#   2) Snap: ADMM tells us which slopes sit exactly at +-L (the active constraints). Fixing those and solving the
#      remaining linear system gives the exact optimum. If it does not verify, we keep the ADMM result.
#   3) Certificate: the multipliers mu give a guaranteed lower bound on the optimum (the dual value). The relative
#      gap between the returned fit and this bound says how close we are to the true floor.


def _diff(v):
    # D v: differences between neighbours, length n-1
    return v[1:] - v[:-1]


def _diff_T(mu):
    # D^T mu: transpose of the difference operator, length n
    return np.concatenate([[0.0], mu]) - np.concatenate([mu, [0.0]])


def lipschitz_floor(x, y, L, gap_tol=1e-5, tol=1e-10, max_iter=50000, return_gap=False):
    """
    Best L-Lipschitz fit to the points (x, y) in 1D.

    Args:
        x: 1D array of inputs (any order, duplicates allowed)
        y: 1D array of targets. Use the noise-free target if you have one, otherwise the floor also contains noise.
        L: Lipschitz constant
        gap_tol: warn if the relative gap to the true optimum is larger than this
        tol: stopping tolerance for ADMM
        max_iter: maximum number of ADMM iterations
        return_gap: also return the relative gap between the returned fit and the dual lower bound

    Returns:
        g: the best L-Lipschitz values at the input points (same order as x). Exactly L-Lipschitz.
        mse: the floor, mean((g - y)^2)
        gap (optional): relative distance to the true optimum (upper bound). Small (< 1e-5) means converged.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same length, got {x.shape} and {y.shape}")
    if L < 0:
        raise ValueError("L must be non-negative")

    # Points with the same x must get the same g, so we merge them. Minimising the squared error of a group
    # is the same as fitting the group mean with weight = group size.
    x_unique, inverse, w = np.unique(x, return_inverse=True, return_counts=True)
    w = w.astype(np.float64)
    y_mean = np.bincount(inverse, weights=y) / w
    n = len(x_unique)
    max_step = L * np.diff(x_unique)
    m = n - 1

    if n == 1 or np.all(np.abs(_diff(y_mean)) <= max_step):
        # The data itself is already L-Lipschitz, so the best fit is the data
        g_unique, gap = y_mean, 0.0
    else:
        # --- 1) ADMM for: min 1/2 sum w (g - y)^2   with   z = D g,  |z| <= max_step
        def system_matrix(rho):
            # (W + rho D^T D) in banded form: one diagonal above and one below the main diagonal
            main = w + rho * (np.r_[0.0, np.ones(m)] + np.r_[np.ones(m), 0.0])
            banded = np.zeros((3, n))
            banded[0, 1:], banded[1], banded[2, :-1] = -rho, main, -rho
            return banded

        rho = 1.0
        banded = system_matrix(rho)
        z = np.clip(_diff(y_mean), -max_step, max_step)
        u = np.zeros(m)
        scale = np.sqrt(np.sum(w * y_mean**2)) + 1e-300
        for it in range(max_iter):
            g = solve_banded((1, 1), banded, w * y_mean + rho * _diff_T(z - u))  # fit the data
            Dg = _diff(g)
            z_old = z
            z = np.clip(Dg + u, -max_step, max_step)                              # keep slopes within +-L
            u = u + Dg - z
            primal_res = np.linalg.norm(Dg - z)
            dual_res = rho * np.linalg.norm(_diff_T(z - z_old))
            if primal_res < tol * scale and dual_res < tol * scale:
                break
            if it % 50 == 0:  # keep the two residuals balanced, this speeds up convergence a lot
                if primal_res > 10 * dual_res:
                    rho, u = 2 * rho, u / 2
                    banded = system_matrix(rho)
                elif dual_res > 10 * primal_res:
                    rho, u = rho / 2, 2 * u
                    banded = system_matrix(rho)
        mu = rho * u  # multipliers: the optimum satisfies g = y - D^T mu / w

        # --- 2) Snap to the exact optimum: fix the active slopes at +-L and solve for the multipliers
        upper = (mu > 0) | (Dg > max_step * (1 - 1e-9))
        lower = (mu < 0) | (Dg < -max_step * (1 - 1e-9))
        active = upper | lower
        if active.any():
            M = diags([-1 / w[1:-1], 1 / w[:-1] + 1 / w[1:], -1 / w[1:-1]], [-1, 0, 1], format="csc")  # D W^-1 D^T
            M_active = M[active][:, active]
            rhs = _diff(y_mean)[active] - np.where(upper[active], max_step[active], -max_step[active])
            mu_snap = np.zeros(m)
            mu_snap[active] = spsolve(M_active, rhs) if active.sum() > 1 else rhs / M_active.toarray().ravel()
            g_snap = y_mean - _diff_T(mu_snap) / w
            verified = (np.all(np.abs(_diff(g_snap)) <= max_step * (1 + 1e-9) + 1e-14)
                        and np.all(mu_snap[upper] >= -1e-12) and np.all(mu_snap[lower] <= 1e-12))
            if verified:
                g, mu = g_snap, mu_snap

        # Remove rounding-level violations: clip the steps to +-L and choose the best constant offset
        steps = np.clip(_diff(g), -max_step, max_step)
        g_shape = np.concatenate([[0.0], np.cumsum(steps)])
        g_unique = g_shape + np.sum(w * (y_mean - g_shape)) / np.sum(w)

        # --- 3) Certificate: gap between this fit (upper bound) and the dual value (lower bound on the optimum)
        DT_mu = _diff_T(mu)
        dual = -(0.5 * np.sum(DT_mu**2 / w) - mu @ _diff(y_mean) + max_step @ np.abs(mu))
        primal = 0.5 * np.sum(w * (g_unique - y_mean) ** 2)
        total = 0.5 * np.sum(w * (y_mean - np.average(y_mean, weights=w)) ** 2)
        gap = max(primal - dual, 0.0) / max(primal, 1e-9 * total, 1e-300)
        if gap > gap_tol:
            warnings.warn(f"lipschitz_floor did not converge (relative gap {gap:.1e}). The floor may be overestimated.")

    g = g_unique[inverse]
    mse = np.mean((g - y) ** 2)
    if return_gap:
        return g, mse, gap
    return g, mse


def lipschitz_floor_scan(x, y, L_values, **kwargs):
    """Floor MSE for a list of Lipschitz constants. Useful for plotting MSE against L (or L / L_true)."""
    return np.array([lipschitz_floor(x, y, L, **kwargs)[1] for L in L_values])
