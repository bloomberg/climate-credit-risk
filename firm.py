import numpy as np
import pandas as pd
from scipy import interpolate, optimize, stats

from utils import SCENARIOS, SECTORS, gauss_legendre


class Firm:
    """
    Single firm optimizing its carbon emission with respect to a benchmark
    emission `e_bench`.
    """

    def __init__(
        self,
        P0: float,
        r: float,
        sig: float,
        a: float,
        b: float,
        c: np.ndarray,
        w_1: float,
        w_2: float,
        n_units: float,
        T_final: float,
        sector: str,
        scenario: str,
        alpha: np.ndarray = np.zeros(1),
        beta: np.ndarray = 0.5 * np.ones(1),
        theta: np.ndarray = np.ones(1),
    ):
        """
        Initialize the set of parameters for the portfolio loss.

        Parameters
        ----------
        P0 : float
            Initial production of the firm, must be positive.
        r : float
            Interest rate, must be positive.
        sig : float
            Volatility modeling uncertainty in demand, must be positive.
        a : float
            Average production level. Constant term in the drift function.
        b : float
            Mean-reverting parameter.
        c : np.ndarray
            Firm's dependence to CO2 emission.
        w_1 : float
            Penalty coefficient.
        w_2 : float
            Reward coefficient.
        n_units : float
            Number of units sold.
        T_final : float
            Final maturity.
        sector : str
            Sector name. Must of one SECTORS.
        scenario : str
            Climate scenario (SSP) name. Must of one SCENARIOS.
        alpha : np.ndarray, optional
            Linear coefficient(s) for cost function, by default np.zeros(1).
        beta : np.ndarray, optional
            Quadratic coefficient(s) for cost function, by default 0.5*np.ones(1).
        theta : np.ndarray, optional
            Inverse of emission factor "energy / CO2e ratio" for each energy e,
            by default np.ones(1).
        """

        if P0 <= 0:
            raise ValueError("Initial production `P0` must be positive.")

        if r <= 0:
            raise ValueError("Interest rate `r` must be positive.")

        if sig <= 0:
            raise ValueError("Volatility `sig` must be positive.")

        if alpha.shape[0] != c.shape[0]:
            raise ValueError("`alpha` and `c` must have the same shape.")

        if beta.shape[0] != c.shape[0]:
            raise ValueError("`beta` and `c` must have the same shape.")

        if theta.shape[0] != c.shape[0]:
            raise ValueError("`theta` and `c` must have the same shape.")

        if scenario not in SCENARIOS:
            raise ValueError(f"`scenario` not in {SCENARIOS}.")

        if sector not in SECTORS:
            raise ValueError(f"`sector` not in {SECTORS}.")

        self.n_energy = c.shape[0]
        self.P0 = P0
        self.r = r
        self.sig = sig
        self.a = a
        self.b = b
        self.c = c
        self.w_1 = w_1
        self.w_2 = w_2
        self.n_units = n_units
        self.T_final = T_final
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.gamma_zero_w = self.gamma_unpenalized()
        self.sector = sector
        self.scenario = scenario

        if self.w_1 == 0.0 and self.w_2 == 0.0:
            # unpenalized scenario
            def f_scenario(t):
                return self.gamma_zero_w * np.ones_like(t)

            self.f_scenario = f_scenario

        # Load SSP data
        df_ssp = pd.read_csv("data/SSP_CMIP6_201811.csv")
        # Filter data for the given scenario and the world region
        df0 = df_ssp[
            (df_ssp["REGION"] == "World")
            & (df_ssp["VARIABLE"] == f"CMIP6 Emissions|CO2|{self.sector}")
        ].reset_index(drop=True)
        df = df0[df0["SCENARIO"].isin(SCENARIOS)][
            [
                "SCENARIO",
                "2015",
                "2020",
                "2030",
                "2040",
                "2050",
                "2060",
                "2070",
                "2080",
                "2090",
                "2100",
            ]
        ]
        self.ssp_values = (
            df[df["SCENARIO"] == self.scenario].iloc[0, 1:].values.astype(float)
        )
        self.ssp_values *= self.gamma_zero_w / self.ssp_values[0]
        # Initial year 2015 corresponds to t=0
        self.ssp_time = np.array(
            [0.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]
        )
        # Interpolatation function for the scenario
        self.f_scenario = interpolate.PchipInterpolator(
            x=self.ssp_time,
            y=self.ssp_values,
        )

    def gamma_unpenalized(self) -> float:
        """
        Return the optimal constant emission for the unpenalized case (w1 = w2 = 0).

        Returns
        -------
        float
            The unpenalized optimal constant emission.
        """
        gamma_zero_w_e = (
            self.c * self.theta / (self.r + self.b) - self.alpha * self.theta
        ) / (2.0 * self.beta * self.theta)

        return float(np.sum(gamma_zero_w_e))

    def gamma_optimal_energy(self, t: np.ndarray) -> np.ndarray:
        """
        Return the optimal emission strategy per energy.

        Parameters
        ----------
        t : np.ndarray
            Time.

        Returns
        -------
        gamma_opt: np.ndarray, shape is (number of energy, shape of 't')
            Optimal emission per energy e.
        """

        c_theta = self.c * self.theta
        alpha_theta = self.alpha * self.theta
        beta_theta = self.beta * self.theta
        xi_1 = self.w_1 / np.sum(beta_theta)
        xi_2 = self.w_2 / np.sum(beta_theta)

        # Compute excess and lack of emissions
        gamma_excess = (2.0 * self.w_1 / (1 + xi_1)) * np.maximum(
            self.gamma_unpenalized() - self.f_scenario(t), 0.0
        )
        gamma_lack = (2.0 * self.w_2 / (1 - xi_2)) * np.maximum(
            self.f_scenario(t) - self.gamma_unpenalized(), 0.0
        )

        # Compute optimal emission per energy
        gamma_opt_per_e = (
            c_theta[:, None] / (self.r + self.b)
            - alpha_theta[:, None]
            - gamma_excess[None, :]
            - gamma_lack[None, :]
        )

        if not np.all(gamma_opt_per_e >= 0):
            raise ValueError("The optimal emission gamma has to be positive.")

        gamma_opt_per_e /= 2.0 * beta_theta[:, None]

        return gamma_opt_per_e

    def gamma_optimal(self, t: np.ndarray) -> np.ndarray:
        """
        Return the firm's optimal emission strategy, which is the sum of
        all optimal emissions per energy.

        Parameters
        ----------
        t : np.ndarray
            Time.

        Returns
        -------
        np.ndarray, shape of 't'
            The firm's optimal emission strategy.
        """
        return self.gamma_optimal_energy(t=t).sum(axis=0)

    def mean_production(self, u: float, t: float, n_leg=20) -> float:
        """
        Return the mean of the firm's production at time u conditionally on time t.

        Parameters
        ----------
        u : np.ndarray
            Time points at which to calculate the mean.

        Returns
        -------
        np.ndarray
            Mean of the firm's production at time u.
        """
        if not u >= t:
            raise ValueError("u must be greater or equal than t.")

        knots, weights = gauss_legendre(a=t, b=u, n=n_leg)
        c_theta_gamma = np.sum(
            (self.c * self.theta)[:, None] * self.gamma_optimal_energy(t=knots),
            axis=0,
        )
        mean_p = (self.a / self.b) * (1.0 - np.exp(-self.b * (u - t)))
        mean_p += np.sum(weights * np.exp(-self.b * (u - knots)) * c_theta_gamma)
        return mean_p

    def variance_production(self, u: np.ndarray) -> np.ndarray:
        """
        Return the variance of the firm's production at time u.

        Parameters
        ----------
        u : np.ndarray
            Time points at which to calculate the variance.

        Returns
        -------
        np.ndarray
            Variance of the firm's production at time u.
        """
        return self.sig**2 * (1.0 - np.exp(-2.0 * self.b * u)) / (2.0 * self.b)

    def optimal_log_prod(self, tab_T, n_mc: int = 1, seed=None):
        """
        Compute the optimal log production.

        Parameters
        ----------
        tab_T: array-like with shape (n_time, 1)
            Time points.
        n_mc : int, optional
            Number of Monte Carlo simulations (default is 100).
        seed : int, optional
            Seed for the random number generator (default is None).

        Returns
        -------
        log_prod : array-like
            Optimal log production at each time point.
        """
        n_disc = tab_T.shape[0]

        if seed is not None:
            np.random.seed(seed)

        Z = np.random.randn(n_disc, n_mc)
        log_prod = np.zeros_like(Z)
        log_prod[0, :] = np.log(self.P0)

        for i in range(n_disc - 1):
            dT_i = tab_T[i + 1] - tab_T[i]
            mean_i = (
                self.mean_production(u=tab_T[i + 1], t=tab_T[i])
                + np.exp(-self.b * dT_i) * log_prod[i, :]
            )
            std_i = np.sqrt(self.variance_production(u=dT_i))
            log_prod[i + 1, :] = mean_i + std_i * Z[i, :]

        return log_prod

    def h(self, t, x, n_leg=20):
        """
        Compute the firm's optimal value using the function `h`.

        The function `h` is defined as:
        h(t, p_t) = V_t with p_t = optimal log production at time `t`.

        Parameters
        ----------
        t: float
            Current time.
        x: float
            Log production.
        n_leg : int, optional
            Number of nodes for Gauss-Legendre quadrature (default is 20).

        Returns
        -------
        opt_value : float or array-like
            Optimal firm's value.
        """
        x = np.atleast_1d(x)
        knots, weights = gauss_legendre(a=t, b=self.T_final, n=n_leg)

        gamma_opt_knots = self.gamma_optimal(knots)
        gamma_opt_e_knots = self.gamma_optimal_energy(knots)
        f_scen_knots = self.f_scenario(knots)

        discount_knots = np.exp(-self.r * (knots - t))
        mean_knots = np.array(
            [self.mean_production(u=knot, t=t, n_leg=n_leg) for knot in knots]
        )
        integrand_mean_cov = self.n_units * np.exp(
            np.exp(-self.b * (knots - t))[:, None] * x[None, :]
            + mean_knots[:, None]
            + 0.5 * self.variance_production(knots - t)[:, None]
        )
        integrand_alpha_beta = np.sum(
            (self.alpha * self.theta)[:, None] * gamma_opt_e_knots
            + (self.beta * self.theta)[:, None] * gamma_opt_e_knots**2,
            axis=0,
        )
        integrand_penal = self.w_1 * np.maximum(gamma_opt_knots - f_scen_knots, 0) ** 2
        integrand_reward = self.w_2 * np.maximum(f_scen_knots - gamma_opt_knots, 0) ** 2

        integrand = (
            integrand_mean_cov
            - integrand_alpha_beta[:, None]
            - integrand_penal[:, None]
            + integrand_reward[:, None]
        )
        optimal_value = np.sum((weights * discount_knots)[:, None] * integrand, axis=0)

        return optimal_value

    # def derivative_h_x(self, t, x, n_leg=20):
    #     x = np.atleast_1d(x)
    #     knots, weights = gauss_legendre(a=t, b=self.T_final, n=n_leg)
    #     mean_knots = np.array(
    #         [self.mean_production(u=knot, t=t, n_leg=n_leg) for knot in knots]
    #     )
    #     discount_knots = np.exp(-self.r * (knots - t))
    #     integrand = (
    #         np.exp(-self.b * (knots - t))[:, None]
    #         * self.n_units
    #         * np.exp(
    #             np.exp(-self.b * (knots - t))[:, None] * x[None, :]
    #             + mean_knots[:, None]
    #             + 0.5 * self.variance_production(knots - t)[:, None]
    #         )
    #     )
    #     return np.sum((weights * discount_knots)[:, None] * integrand, axis=0)

    def proba_default_merton(self, t, L_t, n_leg=20):
        """
        Compute the probability of default P(V_t <= L_t) using Merton's model.

        V_t denotes the optimal's firm value.

        Parameters
        ----------
        t : float
            Time point at which to compute the probability of default.
        L_t : float
            Default barrier at time t.
        n_leg : int, optional
            Number of nodes for Gauss-Legendre quadrature (default is 20).

        Returns
        -------
        prob_default : float
            Probability of default at time t.
        """

        knots, weights = gauss_legendre(a=t, b=self.T_final, n=n_leg)
        gamma_opt_knots = self.gamma_optimal(knots)
        gamma_opt_e_knots = self.gamma_optimal_energy(knots)
        f_scen_knots = self.f_scenario(knots)
        discount_knots = np.exp(-self.r * (knots - t))
        mean_knots = np.array(
            [self.mean_production(u=knot, t=t, n_leg=n_leg) for knot in knots]
        )

        integrand_mean_cov = self.n_units * np.exp(
            mean_knots + 0.5 * self.variance_production(knots - t)
        )
        integrand_alpha_beta = np.sum(
            (self.alpha * self.theta)[:, None] * gamma_opt_e_knots
            + (self.beta * self.theta)[:, None] * gamma_opt_e_knots**2,
            axis=0,
        )
        integrand_penal = self.w_1 * np.maximum(gamma_opt_knots - f_scen_knots, 0) ** 2
        integrand_reward = self.w_2 * np.maximum(f_scen_knots - gamma_opt_knots, 0) ** 2

        integrand_no_x = -integrand_alpha_beta - integrand_penal + integrand_reward
        integral_no_x = np.sum(weights * discount_knots * integrand_no_x)

        def fun(x):
            # minimize function to find the inverse
            integrand_x = integrand_mean_cov * np.exp(np.exp(-self.b * (knots - t)) * x)
            h_x = np.sum(weights * discount_knots * integrand_x) + integral_no_x
            return (h_x - L_t) ** 2

        h_T_inv_L = optimize.minimize_scalar(fun=fun).x
        tmp_ = (
            h_T_inv_L
            - (
                self.mean_production(u=t, t=0.0, n_leg=n_leg)
                + np.exp(-self.b * t) * np.log(self.P0)
            )
        ) / np.sqrt(self.variance_production(t))
        prob_default = stats.norm.cdf(tmp_)

        return prob_default

    @staticmethod
    def default_boundary_unpenalized(firm_no_pen, t, lbd_ref, n_leg=20):
        """
        Compute the default boundary L_T for an unpenalized firm.

        The default boundary is calculated using the formula:
        `P(V_T <= L) = 1 - exp(-lbd_ref * T)`

        Parameters
        ----------
        firm_no_pen : Firm
            Firm instance with zero emission constraint (w1 = w2 = 0).
        t : float
            Time point at which to calculate the default boundary.
        lbd_ref : float
            Reference intensity.
        n_leg : int, optional
            Number of quadrature points to use for the integration (default is 20).

        Returns
        -------
        float
            Default boundary L_T.

        Raises
        ------
        ValueError
            If the firm's penalization coefficient w_1 or reward coefficient
            w_2 is not zero.
        """
        if firm_no_pen.w_1 != 0.0:
            raise ValueError(
                "Penalization coefficient w_1 must be greater or equal to 0."
            )

        if firm_no_pen.w_2 != 0.0:
            raise ValueError("Reward coefficient w_2 must be greater or equal to 0.")

        # params_no_pen = params.copy()
        # params_no_pen.update({"w_1": 0.0, "w_2": 0.0})
        # firm_no_pen = cls(**params_no_pen)

        x = (
            stats.norm.ppf(1.0 - np.exp(-lbd_ref * t))
            * np.sqrt(firm_no_pen.variance_production(t))
            + firm_no_pen.mean_production(u=t, t=0.0, n_leg=n_leg)
            + np.exp(-firm_no_pen.b * t) * np.log(firm_no_pen.P0)
        )
        boundary = firm_no_pen.h(t=t, x=x, n_leg=n_leg)

        return boundary

    @staticmethod
    def default_intensity(tab_t: np.ndarray, prob_default: np.ndarray) -> np.ndarray:
        """
        Compute the intensity for each probability of default.

        The intensity is calculated using the formula:
        lambda_t_{i-1} = log((1-pd_t_{i-1})/(1-pd_t_{i})) / (t_{i} - t_{i-1})

        Parameters
        ----------
        tab_t : np.ndarray
            Time points.
        prob_default : np.ndarray
            Probabilities of default at each time point.

        Returns
        -------
        np.ndarray
            Associated intensity for each probability of default.
        """
        return np.log((1.0 - prob_default[:-1]) / (1.0 - prob_default[1:])) / (
            tab_t[1:] - tab_t[:-1]
        )
