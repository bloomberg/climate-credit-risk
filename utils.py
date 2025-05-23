"""
Utility functions for climate credit risk analysis.

This module provides various mathematical and statistical functions, including:
- Polynomial Chaos Expansion (PCE) computations.
- Gauss-Hermite and Gauss-Legendre quadrature methods.
- Cholesky decomposition using SVD.
- Helper functions for climate scenario analysis.

It also defines constants for plotting and scenario configurations.
"""

import math
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import sparse, special, stats
from tqdm import tqdm

# Colors used in plots
COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
)
# Climate scenarios (SSPs) used in the paper
SCENARIOS = (
    "SSP1-26",
    "SSP2-45",
    "SSP3-70 (Baseline)",
    "SSP4-60",
    "SSP5-85 (Baseline)",
)
# Sectors for each SSP
SECTORS = (
    "Energy Sector",
    "Industrial Sector",
    "Residential Commercial Other",
    "Solvents Production and Application",
    "Transportation Sector",
    "Waste",
    "AFOLU",
    "Aircraft",
    "International Shipping",
)
# Dictionnary associating one color to one SSP
COLORS_SSP = {
    "SSP1-26": "tab:green",
    "SSP2-45": "tab:blue",
    "SSP3-70 (Baseline)": "tab:orange",
    "SSP4-60": "tab:purple",
    "SSP5-85 (Baseline)": "tab:red",
}


def cholesky_from_svd(a: np.ndarray) -> np.ndarray:
    """
    Compute the Cholesky decomposition of a matrix using SVD and QR.

    This function works with positive semi-definite matrices.

    Parameters
    ----------
    a : np.ndarray
        The input matrix.

    Returns
    -------
    np.ndarray
        The Cholesky decomposition of the input matrix.
    """
    u, s, _ = np.linalg.svd(a)
    b = np.diag(np.sqrt(s)) @ u.T
    _, r = np.linalg.qr(b)
    return r.T


def gauss_legendre(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Legendre quadrature points and weights on the interval [a, b].

    Parameters
    ----------
    a : float
        Lower bound of the integration interval.
    b : float
        Upper bound of the integration interval.
    n : int
        Number of quadrature points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two 1-D arrays:
        - Quadrature points on [a, b].
        - Quadrature weights on [a, b].
    """
    knots, weights = np.polynomial.legendre.leggauss(n)
    knots_a_b = 0.5 * (b - a) * knots + 0.5 * (b + a)
    weights_a_b = 0.5 * (b - a) * weights
    return knots_a_b, weights_a_b


def gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Hermite quadrature points and weights.

    Integration is with respect to the Gaussian density. It corresponds to the
    probabilist's Hermite polynomials.

    Parameters
    ----------
    n: int
        Number of quadrature points.

    Returns
    -------
    knots: array-like
        Gauss-Hermite knots.
    weight: array-like
        Gauss-Hermite weights.
    """
    knots, weights = np.polynomial.hermite.hermgauss(n)
    knots *= np.sqrt(2)
    weights /= np.sqrt(np.pi)
    return knots, weights


def coef_gauss_pce(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute the PCE coefficient of the random variable 1_{x < Z} with Z = N(0,1).

    The PCE coefficient is computed as E[ He_n(Z) 1_{x < Z} ] / n! where He_n is
    the nth order Hermite polynomial.

    Parameters
    ----------
    n : int
        The order of the PCE.
    x : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        The nth order PCE coefficient at x.
    """
    if n == 0:
        return stats.norm.cdf(-x)
    coef = stats.norm.pdf(x) * special.hermitenorm(n=n - 1)(x) / math.factorial(n)
    return coef


def mean_gauss_pce(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the mean vector E[gamma_i (a X + b)] for i=1,...,n

    The term gamma_i is the ith PCE coefficient, X = N(0,1), and a, b two vectors.

    Parameters
    ----------
    n : int
        The order of the PCE.
    a : np.ndarray
        Vector of standard deviations.
    b : np.ndarray
        Vector of means.

    Returns
    -------
    np.ndarray, shape is (n+1, shape of a)
        The mean vector for the PCE.

    Raises
    ------
    ValueError
        If a and b do not have the same shape.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have the same shape.")

    mean_pce = np.zeros((n + 1, a.shape[0]), dtype=np.float64)
    x = b / (1.0 + a**2) ** 0.5
    for i in range(n + 1):
        if i == 0:
            mean_pce[0, :] = stats.norm.cdf(-x)
        elif i == 1:
            mean_pce[1, :] = (
                np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi) / np.sqrt(1.0 + a**2)
            )
        elif i == 2:
            mean_pce[2, :] = (b / 2.0) * mean_pce[1, :] / (1.0 + a**2)
        else:
            mean_pce[i, :] = (b / i) * mean_pce[i - 1, :] - (
                (i - 2) / (i * (i - 1.0))
            ) * mean_pce[i - 2, :]
            mean_pce[i, :] /= 1.0 + a**2

    return mean_pce


def _sigma0_matrix(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the first row of the PCE covariance matrix
    for a given order n and input vectors a and b.

    Parameters
    ----------
    n : int
        The order of the PCE.
    a : np.ndarray
        Vector of standard deviations.
    b : np.ndarray
        Vector of means.

    Returns
    -------
    np.ndarray
        The sigma0 matrix of size (n+1, K).
    """

    size_a = np.shape(a)[0]
    sigma0_n = np.zeros((n + 1, size_a), dtype=np.float64)
    mu_n = mean_gauss_pce(n=n, a=a, b=b)
    mu_n2 = mean_gauss_pce(n=n, a=a / np.sqrt(1.0 + a**2), b=b / (1.0 + a**2))

    tmp = np.zeros_like(a)
    for k in range(size_a):
        mean_k = [0, 0]
        cov_k = [[1 + a[k] ** 2, a[k] ** 2], [a[k] ** 2, 1 + a[k] ** 2]]
        tmp[k] = stats.multivariate_normal.cdf(x=[-b[k], -b[k]], mean=mean_k, cov=cov_k)

    sigma0_n[0] = tmp - mu_n[0] ** 2

    if n == 0:
        return sigma0_n

    sigma0_n[1] = mu_n[1] * (mu_n2[0] - mu_n[0])
    if n == 1:
        return sigma0_n

    for i in range(2, n + 1):
        tmp = b * sigma0_n[i - 1, :] - ((i - 2) / (i - 1)) * sigma0_n[i - 2, :]
        tmp -= (a**2) * mu_n[1] * mu_n2[i - 1]
        sigma0_n[i] = tmp / (i * (1 + a**2))

    return sigma0_n


def cov_gauss_pce(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the covariance matrix of a polynomial chaos expansion (PCE).

    The covariance matrix is computed as:
    Cov[gamma_i (a X + b), gamma_j (a X + b)] for i,j = 1,...,n
    where gamma_i is the ith order PCE coefficient, X = N(0,1),
    and a and b are two vectors.

    Parameters
    ----------
    n : int
        Order of the PCE.
    a : np.ndarray
        Vector of standard deviations.
    b : np.ndarray
        Vector of means.

    Returns
    -------
    np.ndarray, shape is (n+1, n+1, shape of a)
        Covariance matrix.

    Raises
    ------
    ValueError
        If a and b do not have the same shape.
    """

    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have the shape.")

    mean_vector = mean_gauss_pce(n=n, a=a, b=b)

    cov_matrix = np.zeros((n + 1, n + 1, a.shape[0]))
    cov_matrix[0, :, :] = _sigma0_matrix(n=n, a=a, b=b)
    cov_matrix[:, 0, :] = cov_matrix[0, :, :]

    for i in range(0, n):
        for j in range(0, n - 1):
            cov_matrix[i + 1, j + 1, :] = (1 / ((i + 1) * (a**2))) * (
                -(1 + a**2) * (j + 2) * cov_matrix[i, j + 2, :]
                + b * cov_matrix[i, j + 1, :]
                - (j / (j + 1)) * cov_matrix[i, j, :]
            ) - mean_vector[i + 1] * mean_vector[j + 1]

    return cov_matrix


def mean_cov_pce_quad(
    a: np.ndarray, b: np.ndarray, n_pce: int, n_quad: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean vector and covariance matrix of a polynomial chaos expansion (PCE)
    using a Gauss-Hermite quadrature.

    Parameters
    ----------
    a : numpy.ndarray
        Coefficients.
    b : numpy.ndarray
        Coefficients.
    n_pce : int
        Order of the PCE.
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    mean_pce_quad : numpy.ndarray
        Mean vector of the PCE.
    cov_pce_quad : numpy.ndarray
        Covariance matrix of the PCE.
    """

    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    def compute_mean_pce(w_he, y_he_ab, sig):
        """
        Compute the mean vector for the PCE using Gauss-Hermite quadrature.
        """
        mean_pce_quad = np.zeros((n_pce + 1, a.shape[0]))
        for m in range(n_pce + 1):
            if m == 0:
                mean_pce_quad[m] = stats.norm.cdf(-b / (1.0 + a**2) ** 0.5)
            elif m == 1:
                mean_pce_quad[m] = (
                    stats.norm.pdf(b / (1.0 + a**2) ** 0.5) / (1.0 + a**2) ** 0.5
                )
            else:
                mean_pce_quad[m] = (
                    np.sum(
                        w_he[:, None] * special.hermitenorm(m - 1)(y_he_ab),
                        axis=0,
                    )
                    * sig
                    * np.exp(-0.5 * b**2 / (a**2 + 1.0))
                    / (np.sqrt(2.0 * np.pi) * math.factorial(m))
                )
        return mean_pce_quad

    def compute_cov_pce(x_he_ab, y_he_ab, w_he, sig, mean_pce_quad):
        """
        Compute the covariance matrix for the PCE using Gauss-Hermite
        quadrature.
        """
        cov_pce_quad = np.zeros((n_pce + 1, n_pce + 1, a.shape[0]))
        for m1 in range(n_pce + 1):
            if m1 == 0:
                integrand_m1 = coef_gauss_pce(n=0, x=x_he_ab)
            else:
                integrand_m1 = (
                    special.hermitenorm(m1 - 1)(y_he_ab)
                    * sig
                    * np.exp(-0.5 * b**2 / (a**2 + 1.0))
                    / (np.sqrt(2.0 * np.pi) * math.factorial(m1))
                )
            for m2 in range(m1, n_pce + 1):
                if m2 == 0:
                    integrand_m2 = coef_gauss_pce(n=0, x=x_he_ab)
                else:
                    integrand_m2 = (
                        special.hermitenorm(m2 - 1)(y_he_ab)
                        * sig
                        * np.exp(-0.5 * b**2 / (a**2 + 1.0))
                        / (np.sqrt(2.0 * np.pi) * math.factorial(m2))
                    )
                cov_pce_quad[m1, m2, :] = np.sum(
                    w_he[:, None] * integrand_m1 * integrand_m2, axis=0
                )
                cov_pce_quad[m1, m2, :] -= mean_pce_quad[m1, :] * mean_pce_quad[m2, :]

        # Fill in the lower triangular part of the covariance matrix
        cov_pce_quad = np.triu(cov_pce_quad) + np.tril(
            cov_pce_quad.transpose(1, 0, 2), -1
        )
        return cov_pce_quad

    x_he, w_he = gauss_hermite(n=n_quad)
    x_he_ab = a[None, :] * x_he[:, None] + b[None, :]
    mu = -a * b / (1.0 + a**2)
    sig = 1.0 / np.sqrt(1.0 + a**2)
    y_he_ab = a[None, :] * (sig[None, :] * x_he[:, None] + mu[None, :]) + b[None, :]

    # Compute the mean vector and covariance matrix
    mean_pce_quad = compute_mean_pce(w_he, y_he_ab, sig)
    cov_pce_quad = compute_cov_pce(x_he_ab, y_he_ab, w_he, sig, mean_pce_quad)

    return (mean_pce_quad, cov_pce_quad)


def eig_cov_x(n, b_min, b_max, t=1.0, seed=None):
    """
    Compute eigenvalues for covariance matrix X with b = Uniform[b_min, b_max]
    and the correlations rho = Uniform[-1, 1] for a given time t.
    """
    if seed is not None:
        np.random.seed(seed)
    b = np.random.uniform(b_min, b_max, size=n)
    b_rep = np.tile(b, (n, 1))
    rho = np.random.uniform(-1, 1, size=n)
    rho_rep = np.tile(rho, (n, 1))
    cov_x = (
        (rho_rep * rho_rep.T) * (1 - np.exp(-(b_rep + b_rep.T) * t)) / (b_rep + b_rep.T)
    )
    _, eig_x, _ = np.linalg.svd(cov_x)
    return eig_x


def compute_inertia(tab_n, n_mc, b_min=0, b_max=1, t=1.0):
    """
    Compute nu_1 / sum(nu_i) and (nu_1 + nu_2) / sum(nu_1)
    where nu_i is the ith eigenvalue of the the covariance matrix X
    with b = Uniform[b_min, b_max] and correlations rho = Uniform[-1, 1]
    for a given time t. Each time, we run n_mc simulations and
    compute 95% Monte Carlo error.
    """
    list_inertia = []
    list_error = []
    for n in tqdm(tab_n):
        inertia_n_1f = np.zeros(n_mc)
        inertia_n_2f = np.zeros(n_mc)
        for i in range(n_mc):
            b = np.random.uniform(b_min, b_max, size=n)
            b = np.tile(b, (n, 1))
            rho = np.random.uniform(-1, 1, size=n)
            rho = np.tile(rho, (n, 1))
            cov_x = (rho * rho.T) * (1 - np.exp(-(b + b.T) * t)) / (b + b.T)
            eig_x, _ = sparse.linalg.eigs(cov_x, k=2)
            eig_x = np.real(eig_x)
            inertia_n_1f[i] = eig_x[0] / np.trace(cov_x)
            inertia_n_2f[i] = (eig_x[0] + eig_x[1]) / np.trace(cov_x)

        list_inertia.append([np.mean(inertia_n_1f), np.mean(inertia_n_2f)])

        list_error.append(
            [
                1.96 * np.std(inertia_n_1f) / np.sqrt(n_mc),
                1.96 * np.std(inertia_n_2f) / np.sqrt(n_mc),
            ]
        )

    return np.array(list_inertia), np.array(list_error)


def lambda_lgd_ead(idx, n_firms, opt=1):
    """
    Compute the Lambda^idx values for the portfolio where
        Lambda = Loss Given Default (LGD) * Exposure at Default (EAD).

    Parameters:
    -----------
    idx : int or array-like
      Index or indices of obligors.
    opt : int, optional
      Option to determine the function used for Lambda^idx:
      - 1: Homogeneous portfolio, Lambda^idx = 1 / sqrt(i).
      - 2: Lambda^idx = ceil(5 * idx / n_firms)^2.
      - Other: Raises a ValueError.

    Returns:
    --------
    float or array-like
      Computed Lambda^idx values.

    Raises:
    -------
    ValueError
      If the `opt` parameter is not 1 or 2.
    """
    if opt not in (1, 2):
        raise ValueError("Check value for opt.")
    if opt == 1:
        return 1 / np.sqrt(idx)

    return (np.ceil(5 * idx / n_firms)) ** 2


def compute_multi_index(n_pce: int):
    """
    Compute the multi-index for PCE coefficients.
    """
    tab_multi_idx = np.array(
        [
            [int(m1), int(m - m1), int(special.comb(m, m1))]
            for m in np.arange(n_pce + 1)
            for m1 in np.arange(m + 1)
        ]
    ).T
    m1 = tab_multi_idx[0, :]
    m2 = tab_multi_idx[1, :]
    m_comb = tab_multi_idx[2, :]
    return m1, m2, m_comb


def compute_loss_pca_pce(
    n_pce: int,
    n_firms: int,
    n_mc: int,
    params: Dict[str, Any],
    **kwargs,
):
    """
    Compute PCA/PCE approximation for the portfolio loss.

    Parameters
    ----------
    n_pce : int
        Order of the PCE.
    n_firms : int
        Number of firms.
    n_mc : int
        Number of Monte Carlo simulations.
    params : Dict[str, Any]
        Dictionary containing required parameters like l1_normalized, l2_normalized, etc.
    **kwargs : dict
        Optional arguments:
        - return_mean_cov_eps (bool): Whether to return mean and covariance of eps.
        - return_eps_full (bool): Whether to return the full eps vector.
        - return_loss_full (bool): Whether to return the full loss vector.
        - n_quad (int): Number of quadrature points (default: 40).
        - seed (int): Random seed for reproducibility (default: None).

    Returns
    -------
    Depends on the optional arguments provided.
    """
    # Extract optional arguments with defaults

    if kwargs.get("seed", None) is not None:
        np.random.seed(kwargs.get("seed", None))

    tab_m1, tab_m2, tab_multi = compute_multi_index(n_pce)
    l1_normalized_m1 = np.array(
        [params["l1_normalized"][i] ** tab_m1 for i in range(n_firms)]
    ).T
    l2_normalized_m2 = np.array(
        [params["l2_normalized"][i] ** tab_m2 for i in range(n_firms)]
    ).T

    if kwargs.get("return_eps_full", False) or kwargs.get("return_loss_full", False):

        def simulate_vec_eps_full():
            # Simulating vector eps without Gaussian approximation
            vec_a = params["mean_a_normalized"][:, None] + params["std_a_normalized"][
                :, None
            ] * np.random.randn(n_firms, n_mc)
            tau_vec_a_m1_m2 = np.array(
                [coef_gauss_pce(n=int(m), x=vec_a) for m in np.arange(n_pce + 1)]
            )[tab_m1 + tab_m2, :, :]

            # Summing over all firms
            vec_eps_full = np.sum(
                params["tab_lambda"][None, :, None]
                * tab_multi[:, None, None]
                * tau_vec_a_m1_m2
                * l1_normalized_m1[:, :, None]
                * l2_normalized_m2[:, :, None],
                axis=1,
            )
            return vec_eps_full

        vec_eps_full = simulate_vec_eps_full()

        if kwargs.get("return_eps_full", False):
            return vec_eps_full

    # Product of Hermite polynomials
    he_m1_m2 = np.array(
        [
            special.hermitenorm(tab_m1[i])(np.random.randn(n_mc))
            * special.hermitenorm(tab_m2[i])(np.random.randn(n_mc))
            for i in range(int((n_pce + 1) * (n_pce + 2) / 2))
        ]
    )

    if kwargs.get("return_loss_full", False):
        return np.sum(vec_eps_full * he_m1_m2, axis=0)

    def get_cov_eps():
        """Compute the mean and covariance matrix of the vector eps."""
        _mean_a_pce, _cov_a_pce = mean_cov_pce_quad(
            a=params["std_a_normalized"],
            b=params["mean_a_normalized"],
            n_pce=n_pce,
            n_quad=kwargs.get("n_quad", 40),
        )

        # Mean of the vector eps
        mean_eps = np.sum(
            tab_multi[:, None]
            * params["tab_lambda"][None, :]
            * _mean_a_pce[tab_m1 + tab_m2, :]
            * l1_normalized_m1
            * l2_normalized_m2,
            axis=1,
        )

        n_eps = int((n_pce + 1) * (n_pce + 2) / 2)  # shape of the vector eps
        # Covariance of the vector eps
        mat_m1 = np.tile(tab_m1, (n_eps, 1))
        mat_m2 = np.tile(tab_m2, (n_eps, 1))
        mat_m1_m2 = np.tile(tab_m1 + tab_m2, (n_eps, 1))
        mat_l1_normalized = np.array(
            [params["l1_normalized"][i] ** mat_m1 for i in range(n_firms)]
        ).T
        mat_l1_normalized_transpose = np.array(
            [params["l1_normalized"][i] ** mat_m1.T for i in range(n_firms)]
        ).T
        mat_l2_normalized = np.array(
            [params["l2_normalized"][i] ** mat_m2 for i in range(n_firms)]
        ).T
        mat_l2_normalized_transpose = np.array(
            [params["l2_normalized"][i] ** mat_m2.T for i in range(n_firms)]
        ).T

        mat_multi = np.tile(tab_multi, (n_eps, 1))
        cov_eps = (
            mat_multi
            * mat_multi.T
            * np.sum(
                params["tab_lambda"][None, None, :] ** 2
                * _cov_a_pce[mat_m1_m2, mat_m1_m2.T, :]
                * mat_l1_normalized
                * mat_l2_normalized
                * mat_l1_normalized_transpose
                * mat_l2_normalized_transpose,
                axis=2,
            )
        )
        return mean_eps, cov_eps

    if kwargs.get("return_mean_cov_eps", False):
        return get_cov_eps()

    def simulate_vec_eps(n_eps, n_mc, mean_eps, cov_eps):
        """Simulate the vector eps."""
        try:
            # Cholesky
            chol_cov_eps = np.linalg.cholesky(cov_eps)
            z = np.random.randn(n_eps, n_mc)
            vec_eps = mean_eps[:, None] + chol_cov_eps @ z
        except np.linalg.LinAlgError:
            # # Cholesky from SVD
            # print("Cholesky from SVD")
            # chol_cov_eps = cholesky_from_svd(cov_eps)

            # PCA decomposition
            u_cov_eps, eigs_cov_eps, _ = np.linalg.svd(cov_eps)
            vec_eps = np.zeros((eigs_cov_eps.shape[0], n_mc))
            for idx in range(n_eps):
                z = np.random.randn(n_mc)
                u_z = u_cov_eps[:, idx][:, None] * z[None, :]
                vec_eps += np.sqrt(eigs_cov_eps[idx]) * u_z
            vec_eps += mean_eps[:, None]
        return vec_eps

    return np.sum(
        simulate_vec_eps(int((n_pce + 1) * (n_pce + 2) / 2), n_mc, *get_cov_eps())
        * he_m1_m2,
        axis=0,
    )


def set_plot_style() -> None:
    """Set default matplotlib parameters."""
    plt.rcParams["figure.figsize"] = [9.0, 7.0]
    sns.set_style(
        style="ticks",
        rc={
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )
    sns.set_context(
        context="poster",
        rc={
            "grid.linewidth": 1.0,
            "legend.fontsize": "x-small",
            "legend.title_fontsize": "xx-small",
        },
    )
