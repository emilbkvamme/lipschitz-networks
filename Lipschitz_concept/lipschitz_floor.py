# The floor: the best possible L-Lipschitz fit to a target function, i.e. the lowest MSE an L-Lipschitz network can
# reach on the given points. Used by Toy_gaussian.ipynb (1D) and, through Toy_nd_utils.py, by Toy_gaussian_2d.ipynb.
#
# In n dimensions the Lipschitz condition is taken in the L1 norm of the input, |g(x) - g(x')| <= L ||x - x'||_1,
# which is the norm the LipschitzModel networks are bounded in (every partial derivative is at most L).
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def _difference_matrix(axes):
    """
    Sparse matrix D of the differences between neighbouring grid points divided by their distance, for every axis
    stacked on top of each other: (D g) are the slopes along the grid lines. The axes do not need to be evenly spaced.
    """
    shape = [len(a) for a in axes]
    blocks = []
    for i, axis in enumerate(axes):
        m = len(axis)
        first_difference = sp.diags(1 / np.diff(axis)) @ sp.diags([-np.ones(m - 1), np.ones(m - 1)], [0, 1], shape=(m - 1, m))
        factors = [first_difference if j == i else sp.identity(shape[j]) for j in range(len(shape))]
        block = factors[0]
        for factor in factors[1:]:
            block = sp.kron(block, factor)
        blocks.append(block)
    return sp.vstack(blocks).tocsc()


def lipschitz_floor_nd(y, axes, L, tol=1e-9, max_iter=100, return_info=False):
    """
    Best L-Lipschitz fit (in the L1 norm) to the values y on a rectangular n-dimensional grid:

        minimise  sum (g - y)^2   subject to   |g(x) - g(x')| <= L ||x - x'||_1   for all grid points x, x'

    The L1 distance between two grid points is the length of the shortest path along the grid lines, so it is enough
    to constrain neighbours along every axis, |g_(i+1) - g_i| <= L * distance. This is exact, not an approximation, and
    any such grid function can be extended to an L-Lipschitz function on the whole space. In 1D it is the usual floor.

    Solved as a quadratic program with a primal-dual interior point method (Mehrotra predictor-corrector). Every
    iteration solves one sparse system (I + D^T W D) dg = rhs, and it typically converges in 10-40 iterations.

    y: array of shape (m_1, ..., m_n), axes: the grid coordinates along every axis (increasing).
    Returns (floor, MSE), or (floor, MSE, info) with return_info=True.
    """
    y = np.asarray(y, dtype=float)
    shape = y.shape
    y_flat = y.ravel()
    D = _difference_matrix(axes)
    DT = D.T.tocsc()
    identity = sp.identity(y.size, format="csc")
    scale = max(np.max(np.abs(y)), 1e-300)

    # Constraints D g <= L and -D g <= L, with slacks s1, s2 > 0 and multipliers z1, z2 > 0
    g = y_flat.copy()
    s1 = np.full(D.shape[0], float(L))
    s2 = s1.copy()
    z1 = np.full(D.shape[0], scale)
    z2 = z1.copy()
    n_constraints = 2 * D.shape[0]

    def max_step(v, dv):
        """Largest step a <= 1 that keeps v + a dv >= 0."""
        negative = dv < 0
        return min(1.0, np.min(-v[negative] / dv[negative])) if np.any(negative) else 1.0

    for iteration in range(1, max_iter + 1):
        d = D @ g
        r_dual = g - y_flat + DT @ (z1 - z2)
        r1 = d + s1 - L
        r2 = -d + s2 - L
        mu = (s1 @ z1 + s2 @ z2) / n_constraints

        converged = (np.linalg.norm(r_dual) <= tol * scale * np.sqrt(y.size)
                     and max(np.max(np.abs(r1)), np.max(np.abs(r2))) <= tol * L
                     and mu <= tol * scale * L)
        if converged:
            break

        w1, w2 = z1 / s1, z2 / s2
        solve = splu(identity + DT @ sp.diags(w1 + w2) @ D, permc_spec="MMD_AT_PLUS_A").solve

        def newton_step(rc1, rc2):
            """Newton direction for the complementarity residuals rc1 = s1 z1 - target, rc2 = s2 z2 - target."""
            q = (-rc1 + z1 * r1) / s1 - (-rc2 + z2 * r2) / s2
            dg = solve(-r_dual - DT @ q)
            Ddg = D @ dg
            ds1, ds2 = -r1 - Ddg, -r2 + Ddg
            dz1, dz2 = (-rc1 - z1 * ds1) / s1, (-rc2 - z2 * ds2) / s2
            return dg, ds1, ds2, dz1, dz2

        # Predictor: pure Newton step towards mu = 0
        dg, ds1, ds2, dz1, dz2 = newton_step(s1 * z1, s2 * z2)
        step = min(max_step(s1, ds1), max_step(s2, ds2), max_step(z1, dz1), max_step(z2, dz2))
        mu_affine = ((s1 + step * ds1) @ (z1 + step * dz1) + (s2 + step * ds2) @ (z2 + step * dz2)) / n_constraints
        centering = (mu_affine / mu) ** 3

        # Corrector: aim at centering * mu, with the second order term of the predictor
        dg, ds1, ds2, dz1, dz2 = newton_step(s1 * z1 + ds1 * dz1 - centering * mu,
                                             s2 * z2 + ds2 * dz2 - centering * mu)
        step = 0.99 * min(max_step(s1, ds1), max_step(s2, ds2), max_step(z1, dz1), max_step(z2, dz2))
        g, s1, s2, z1, z2 = g + step * dg, s1 + step * ds1, s2 + step * ds2, z1 + step * dz1, z2 + step * dz2

    floor = g.reshape(shape)
    mse = np.mean((floor - y) ** 2)
    if return_info:
        violation = max(np.max(np.abs(D @ g)) - L, 0.0) / L
        return floor, mse, {"iterations": iteration, "converged": converged, "max_violation": violation}
    return floor, mse


def lipschitz_floor(x, y, L):
    """
    Best L-Lipschitz fit to the points (x, y) in 1D. x and y can be numpy arrays or torch tensors of shape (N,) or
    (N, 1), and x does not need to be sorted or evenly spaced.
    Returns the floor as a numpy array of shape (N,), in the same order as x, and its MSE to y.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    order = np.argsort(x)
    floor_sorted, mse = lipschitz_floor_nd(y[order], [x[order]], L)
    floor = np.empty_like(floor_sorted)
    floor[order] = floor_sorted
    return floor, mse


def lipschitz_floor_scan(x, y, L_values):
    """MSE of the 1D floor for every L in L_values, as a numpy array."""
    return np.array([lipschitz_floor(x, y, L)[1] for L in L_values])
