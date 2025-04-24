import math
import time
import logging
import functools
import numpy as np
import torch
import pandas as pd
import datetime
from typing import Tuple, Dict

import plotly.graph_objects as go

try:
    from pandas_datareader import data as web
except ImportError:
    web = None  







logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_execution(func):
    """Decorator to log the execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Starting {func.__name__}...")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"Finished {func.__name__} in {end - start:.3f} seconds.")
        return result
    return wrapper





def cheap_stack(tensors, dim: int):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    else:
        return torch.stack(tensors, dim=dim)

def tridiagonal_solve(b, A_upper, A_diagonal, A_lower):
    """
    Solves a tridiagonal system Ax = b.
    The arguments A_upper, A_diagonal, A_lower correspond to the three diagonals of A.
    """
    A_upper, _ = torch.broadcast_tensors(A_upper, b[..., :-1])
    A_lower, _ = torch.broadcast_tensors(A_lower, b[..., :-1])
    A_diagonal, b = torch.broadcast_tensors(A_diagonal, b)



    channels = b.size(-1)
    new_b = np.empty(channels, dtype=object)
    new_A_diagonal = np.empty(channels, dtype=object)
    outs = np.empty(channels, dtype=object)

    new_b[0] = b[..., 0]
    new_A_diagonal[0] = A_diagonal[..., 0]
    for i in range(1, channels):
        w = A_lower[..., i - 1] / new_A_diagonal[i - 1]
        new_A_diagonal[i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
        new_b[i] = b[..., i] - w * new_b[i - 1]

    outs[channels - 1] = new_b[channels - 1] / new_A_diagonal[channels - 1]
    for i in range(channels - 2, -1, -1):
        outs[i] = (new_b[i] - A_upper[..., i] * outs[i + 1]) / new_A_diagonal[i]

    return torch.stack(outs.tolist(), dim=-1)







def _validate_input(t: torch.Tensor, X: torch.Tensor) -> None:




    if not t.is_floating_point():
        raise ValueError("t must be floating point.")
    if not X.is_floating_point():
        raise ValueError("X must be floating point.")
    if len(t.shape) != 1:
        raise ValueError("t must be one dimensional. It instead has shape {}.".format(tuple(t.shape)))
    prev_t_i = -math.inf
    for t_i in t:
        if t_i <= prev_t_i:
            raise ValueError("t must be monotonically increasing.")
        prev_t_i = t_i
    if X.ndimension() < 2:
        raise ValueError("X must have at least two dimensions (time and channels). It has shape {}.".format(tuple(X.shape)))
    if X.size(-2) != t.size(0):
        raise ValueError("The time dimension of X must equal the length of t.")
    if t.size(0) < 2:
        raise ValueError("Must have a time dimension of size at least 2.")





def _natural_cubic_spline_coeffs_without_missing_values(t: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


    length = x.size(-1)
    if length < 2:
        raise ValueError("Time dimension must be at least 2.")
    elif length == 2:
        a = x[..., :1]
        b = (x[..., 1:] - x[..., :1]) / (t[1:] - t[:1])
        two_c = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
        three_d = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)


    else:
        time_diffs = t[1:] - t[:-1]
        time_diffs_reciprocal = time_diffs.reciprocal()
        time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2
        three_path_diffs = 3 * (x[..., 1:] - x[..., :-1])
        six_path_diffs = 2 * three_path_diffs
        path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared



        system_diagonal = torch.empty(length, dtype=x.dtype, device=x.device)
        system_diagonal[:-1] = time_diffs_reciprocal
        system_diagonal[-1] = 0
        system_diagonal[1:] += time_diffs_reciprocal
        system_diagonal *= 2
        system_rhs = torch.empty_like(x)
        system_rhs[..., :-1] = path_diffs_scaled
        system_rhs[..., -1] = 0
        system_rhs[..., 1:] += path_diffs_scaled
        knot_derivatives = tridiagonal_solve(system_rhs, time_diffs_reciprocal, system_diagonal, time_diffs_reciprocal)




        a = x[..., :-1]
        b = knot_derivatives[..., :-1]
        two_c = (six_path_diffs * time_diffs_reciprocal - 4 * knot_derivatives[..., :-1] - 2 * knot_derivatives[..., 1:]) * time_diffs_reciprocal
        three_d = (-six_path_diffs * time_diffs_reciprocal + 3 * (knot_derivatives[..., :-1] + knot_derivatives[..., 1:])) * time_diffs_reciprocal_squared

    return a, b, two_c, three_d





def _natural_cubic_spline_coeffs_with_missing_values(t: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:



    if x.ndimension() == 1:
        return _natural_cubic_spline_coeffs_with_missing_values_scalar(t, x)
    else:
        a_pieces, b_pieces, two_c_pieces, three_d_pieces = [], [], [], []
        for p in x.unbind(dim=0):
            a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, p)
            a_pieces.append(a)
            b_pieces.append(b)
            two_c_pieces.append(two_c)
            three_d_pieces.append(three_d)
        return (cheap_stack(a_pieces, dim=0),
                cheap_stack(b_pieces, dim=0),
                cheap_stack(two_c_pieces, dim=0),
                cheap_stack(three_d_pieces, dim=0))




def _natural_cubic_spline_coeffs_with_missing_values_scalar(t: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    not_nan = ~torch.isnan(x)
    path_no_nan = x.masked_select(not_nan)
    if path_no_nan.size(0) == 0:
        return (torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device))
    need_new_not_nan = False
    if torch.isnan(x[0]):
        if not need_new_not_nan:
            x = x.clone()
            need_new_not_nan = True
        x[0] = path_no_nan[0]
    if torch.isnan(x[-1]):
        if not need_new_not_nan:
            x = x.clone()
            need_new_not_nan = True
        x[-1] = path_no_nan[-1]
    if need_new_not_nan:
        not_nan = ~torch.isnan(x)
        path_no_nan = x.masked_select(not_nan)
    times_no_nan = t.masked_select(not_nan)



    a_pieces_no_nan, b_pieces_no_nan, two_c_pieces_no_nan, three_d_pieces_no_nan = _natural_cubic_spline_coeffs_without_missing_values(times_no_nan, path_no_nan)

    a_pieces, b_pieces, two_c_pieces, three_d_pieces = [], [], [], []
    iter_times_no_nan = iter(times_no_nan)
    iter_coeffs_no_nan = iter(zip(a_pieces_no_nan, b_pieces_no_nan, two_c_pieces_no_nan, three_d_pieces_no_nan))
    next_time_no_nan = next(iter_times_no_nan)
    for time in t[:-1]:
        if time >= next_time_no_nan:
            prev_time_no_nan = next_time_no_nan
            next_time_no_nan = next(iter_times_no_nan)
            next_a_no_nan, next_b_no_nan, next_two_c_no_nan, next_three_d_no_nan = next(iter_coeffs_no_nan)
        offset = prev_time_no_nan - time
        a_inner = (0.5 * next_two_c_no_nan - next_three_d_no_nan * offset / 3) * offset
        a_pieces.append(next_a_no_nan + (a_inner - next_b_no_nan) * offset)
        b_pieces.append(next_b_no_nan + (next_three_d_no_nan * offset - next_two_c_no_nan) * offset)
        two_c_pieces.append(next_two_c_no_nan - 2 * next_three_d_no_nan * offset)
        three_d_pieces.append(next_three_d_no_nan)




    return (cheap_stack(a_pieces, dim=0),
            cheap_stack(b_pieces, dim=0),
            cheap_stack(two_c_pieces, dim=0),
            cheap_stack(three_d_pieces, dim=0))




def natural_cubic_spline_coeffs(t: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the coefficients for the natural cubic spline approximation.
    Arguments:
        t: 1D tensor of times (must be monotonically increasing).
        x: Tensor of values of shape (..., length, input_channels); missing values should be NaN.
    Returns:
        A tuple (t, a, b, c, d) for use with the NaturalCubicSpline class.
    """
    _validate_input(t, x)
    if torch.isnan(x).any():
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, x.transpose(-1, -2))
    else:
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_without_missing_values(t, x.transpose(-1, -2))
    a = a.transpose(-1, -2)
    b = b.transpose(-1, -2)
    c = two_c.transpose(-1, -2) / 2
    d = three_d.transpose(-1, -2) / 3
    return t, a, b, c, d

class NaturalCubicSpline:
    


    def __init__(self, coeffs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], **kwargs):
        t, a, b, c, d = coeffs
        self._t = t
        self._a = a
        self._b = b
        self._c = c
        self._d = d



    def _interpret_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        maxlen = self._b.size(-2) - 1
        index = torch.bucketize(t.detach(), self._t) - 1
        index = index.clamp(0, maxlen)
        fractional_part = t - self._t[index]
        return fractional_part, index



    def evaluate(self, t: torch.Tensor) -> torch.Tensor:
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = self._c[..., index, :] + self._d[..., index, :] * fractional_part
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part



    def derivative(self, t: torch.Tensor, order: int = 1) -> torch.Tensor:
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        if order == 1:
            inner = 2 * self._c[..., index, :] + 3 * self._d[..., index, :] * fractional_part
            deriv = self._b[..., index, :] + inner * fractional_part
        elif order == 2:
            deriv = 2 * self._c[..., index, :] + 6 * self._d[..., index, :] * fractional_part
        else:
            raise ValueError("Derivative is not implemented for orders greater than 2.")
        return deriv

__version__ = "0.0.3"




class YieldCurveSplineFitter:
    """
    A class to fetch U.S. Treasury yield curve data from FRED, fit a natural cubic spline interpolation,
    a Gaussian Process Regression (GPR) model and a Multi-Layer Perceptron (MLP) regressor,
    and plot the results.
    The data are fetched for a set of U.S. Treasury yields including the following maturities:
      6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr, 20 Yr, and 30 Yr.
    """
    def __init__(self, start_date: datetime.date, end_date: datetime.date):
        self.start_date = start_date
        self.end_date = end_date
        self.data = None            # DataFrame with the yield curve
        self.maturities = None      # 1D numpy array of maturities (in years)
        self.yields = None          # Corresponding yield values (in percentages)
        self.spline = None
        self.gpr_model = None
        self.mlp_model = None

    @log_execution
    def fetch_yield_curve(self) -> None:
        """
        Fetch U.S. Treasury yield curve data from FRED.
        The method collects data for a set of standard maturities using their FRED series codes.
        If fetching a specific series fails, that series is eliminated.
        """
        yield_codes: Dict[str, float] = {
            "DGS6MO": 0.5,
            "DGS1": 1,
            "DGS2": 2,
            "DGS3": 3,
            "DGS5": 5,
            "DGS7": 7,
            "DGS10": 10,
            "DGS20": 20,
            "DGS30": 30
        }





        fetched_data = {}
        if web is not None:
            for code, maturity in yield_codes.items():
                try:
                    df = web.DataReader(code, "fred", self.start_date, self.end_date)
                    df.dropna(inplace=True)
                    if not df.empty:
                        value = float(df.iloc[-1, 0])
                        fetched_data[maturity] = value
                        logging.info(f"Fetched {code}: {value} for maturity {maturity} years.")
                    else:
                        logging.warning(f"Empty data for {code}, skipping series.")
                except Exception as e:
                    logging.warning(f"Could not fetch data for {code}: {e} - Eliminating series.")


        else:
            logging.warning("pandas_datareader not available. Using simulated data.")
            fetched_data = {
                0.5: 0.10,
                1: 0.15,
                2: 0.30,
                3: 0.40,
                5: 0.70,
                7: 0.85,
                10: 1.00,
                20: 1.50,
                30: 1.75
            }


        if not fetched_data:
            logging.warning("No U.S. Treasury data could be fetched from FRED. Using simulated data.")
            fetched_data = {
                0.5: 0.10,
                1: 0.15,
                2: 0.30,
                3: 0.40,
                5: 0.70,
                7: 0.85,
                10: 1.00,
                20: 1.50,
                30: 1.75
            }




        maturities_sorted = np.array(sorted(fetched_data.keys()))
        yields_sorted = np.array([fetched_data[m] for m in maturities_sorted])
        self.maturities = maturities_sorted
        self.yields = yields_sorted
        self.data = pd.DataFrame({"Maturity": self.maturities, "Yield": self.yields})
        logging.info("U.S. Treasury yield curve data fetched successfully.")



    @log_execution
    def fit_spline(self) -> None:
        """
        Fit a natural cubic spline to the yield curve data.
        The independent variable is the maturity (in years) and the dependent variable is the yield.
        """
        if self.data is None:
            raise RuntimeError("Yield curve data not available. Call fetch_yield_curve() first.")
        t = torch.tensor(self.data["Maturity"].values, dtype=torch.float64)
        x = torch.tensor(self.data["Yield"].values, dtype=torch.float64).unsqueeze(-1)
        coeffs = natural_cubic_spline_coeffs(t, x)
        self.spline = NaturalCubicSpline(coeffs)
        logging.info("Cubic spline fitting completed.")



    @log_execution
    def fit_gpr(self) -> None:
       
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.preprocessing import StandardScaler

        if self.data is None:
            raise RuntimeError("Yield curve data not available. Call fetch_yield_curve() first.")
        X = self.maturities.reshape(-1, 1)
        y = self.yields
        scaler_X = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                 + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gpr.fit(X_scaled, y_scaled)
        self.gpr_model = gpr
        self.gpr_scaler_X = scaler_X
        self.gpr_scaler_y = scaler_y
        logging.info(f"GPR fitting completed. Learned kernel: {gpr.kernel_}")

    @log_execution
    def fit_mlp(self) -> None:
        """
        Fit a Multi-Layer Perceptron (MLP) regressor to the yield curve data.
        Feature scaling is applied to both inputs and outputs. The MLP uses two hidden layers,
        and its architecture is designed to capture nonlinear relationships in the data.
        """
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        if self.data is None:
            raise RuntimeError("Yield curve data not available. Call fetch_yield_curve() first.")
        X = self.maturities.reshape(-1, 1)
        y = self.yields
        scaler_X = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()
        mlp = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=10000, random_state=42)
        mlp.fit(X_scaled, y_scaled)
        self.mlp_model = mlp
        self.mlp_scaler_X = scaler_X
        self.mlp_scaler_y = scaler_y
        logging.info("MLP fitting completed.")

    @log_execution
    def plot(self) -> None:
        """
        Plot the original yield curve data, the cubic spline interpolation, the GPR interpolation,
        and the MLP interpolation.
        """
        if self.spline is None:
            raise RuntimeError("Spline not fitted. Call fit_spline() first.")
        t_min, t_max = self.maturities[0], self.maturities[-1]
        t_fine = torch.linspace(t_min, t_max, 200, dtype=torch.float64)
        spline_vals = self.spline.evaluate(t_fine).squeeze().detach().numpy()



        fig = go.Figure()
        # Plot original data points
        fig.add_trace(go.Scatter(x=self.maturities, y=self.yields, mode='markers', name='Yield Curve Data',
                                 marker=dict(size=10, color='red')))
        # Plot cubic spline interpolation
        fig.add_trace(go.Scatter(x=t_fine.detach().numpy(), y=spline_vals, mode='lines', name='Cubic Spline',
                                 line=dict(color='blue', width=2)))
        # Plot GPR interpolation with confidence interval
        if self.gpr_model is not None:
            X_fine = t_fine.detach().numpy().reshape(-1, 1)
            X_fine_scaled = self.gpr_scaler_X.transform(X_fine)
            y_pred_scaled, sigma = self.gpr_model.predict(X_fine_scaled, return_std=True)
            y_pred = self.gpr_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            sigma = sigma * self.gpr_scaler_y.scale_[0]
            fig.add_trace(go.Scatter(x=t_fine.detach().numpy().flatten(), y=y_pred,
                                 mode='lines', name='GPR Interpolation',
                                 line=dict(color='green', dash='dash', width=2)))
            

            fig.add_trace(go.Scatter(
                x=np.concatenate([t_fine.detach().numpy(), t_fine.detach().numpy()[::-1]]),
                y=np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
                fill='toself',
                fillcolor='rgba(0, 255, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='GPR Confidence Interval'
            ))



        if self.mlp_model is not None:
            X_fine = t_fine.detach().numpy().reshape(-1, 1)
            X_fine_scaled = self.mlp_scaler_X.transform(X_fine)
            y_mlp_scaled = self.mlp_model.predict(X_fine_scaled)
            y_mlp = self.mlp_scaler_y.inverse_transform(y_mlp_scaled.reshape(-1, 1)).ravel()
            fig.add_trace(go.Scatter(x=t_fine.detach().numpy().flatten(), y=y_mlp,
                                 mode='lines', name='MLP Interpolation',
                                 line=dict(color='purple', dash='dot', width=2)))
        fig.update_layout(title='U.S. Treasury Yield Curve: Cubic Spline vs. GPR vs. MLP Interpolation',
                          xaxis_title='Maturity (years)',
                          yaxis_title='Yield (%)',
                          template='plotly_white')
        fig.show()





@log_execution
def main():
    start_date = datetime.date.today() - datetime.timedelta(days=60)
    end_date = datetime.date.today()
    fitter = YieldCurveSplineFitter(start_date, end_date)
    fitter.fetch_yield_curve()
    fitter.fit_spline()
    fitter.fit_gpr()
    fitter.fit_mlp()
    fitter.plot()



if __name__ == "__main__":
    main()
