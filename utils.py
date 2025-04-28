import math

import numpy as np
from scipy import special, stats

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


def cholesky_from_svd(A: np.ndarray) -> np.ndarray:
    """
    Compute the Cholesky decomposition of a matrix using SVD and QR.

    This function works with positive semi-definite matrices.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.

    Returns
    -------
    np.ndarray
        The Cholesky decomposition of the input matrix.
    """
    U, S, _ = np.linalg.svd(A)
    B = np.diag(np.sqrt(S)) @ U.T
    _, R = np.linalg.qr(B)
    L = R.T
    return L


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
    else:
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


def mean_cov_pce_quad(a, b, n_pce: int, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
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

    # Gauss-Hermite quadrature
    x_he, w_he = gauss_hermite(n=n_quad)
    x_he_ab = a[None, :] * x_he[:, None] + b[None, :]
    mu = -a * b / (1.0 + a**2)
    sig = 1.0 / np.sqrt(1.0 + a**2)
    y_he_ab = a[None, :] * (sig[None, :] * x_he[:, None] + mu[None, :]) + b[None, :]

    # Mean vector
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

    # Covariance matrix
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
    cov_pce_quad = np.triu(cov_pce_quad) + np.tril(cov_pce_quad.T, -1)

    return (mean_pce_quad, cov_pce_quad)
