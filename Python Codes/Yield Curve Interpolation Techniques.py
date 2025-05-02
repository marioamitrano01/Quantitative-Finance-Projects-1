import math
import time
import logging
import functools
import datetime
from typing import Tuple, Dict, List, Optional, Union, Any, cast
from dataclasses import dataclass

import numpy as np
import torch
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    from pandas_datareader import data as web
    HAS_DATAREADER = True
except ImportError:
    web = None
    HAS_DATAREADER = False
    logger.warning("pandas_datareader not installed. Data fetching from FRED will be unavailable.")
    
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed. GPR and MLP modeling will be unavailable.")


def log_execution(func):
    """Decorator to log the execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}...")
        start = time.time()
        try:
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(f"Finished {func.__name__} in {end - start:.3f} seconds.")
            return result
        except Exception as e:
            end = time.time()
            logger.error(f"Error in {func.__name__} after {end - start:.3f} seconds: {str(e)}")
            raise
    return wrapper


def cache_result(func):
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper

def cheap_stack(tensors: List[torch.Tensor], dim: int) -> torch.Tensor:
    
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    else:
        return torch.stack(tensors, dim=dim)


def tridiagonal_solve(
    b: torch.Tensor, 
    A_upper: torch.Tensor, 
    A_diagonal: torch.Tensor, 
    A_lower: torch.Tensor
) -> torch.Tensor:
    
    A_upper, _ = torch.broadcast_tensors(A_upper, b[..., :-1])
    A_lower, _ = torch.broadcast_tensors(A_lower, b[..., :-1])
    A_diagonal, b = torch.broadcast_tensors(A_diagonal, b)

    channels = b.size(-1)
    
    new_diag = torch.empty_like(A_diagonal)
    new_b = torch.empty_like(b)
    
    new_diag[..., 0] = A_diagonal[..., 0]
    new_b[..., 0] = b[..., 0]
    
    for i in range(1, channels):
        w = A_lower[..., i-1] / new_diag[..., i-1]
        new_diag[..., i] = A_diagonal[..., i] - w * A_upper[..., i-1]
        new_b[..., i] = b[..., i] - w * new_b[..., i-1]
    
    x = torch.empty_like(b)
    x[..., -1] = new_b[..., -1] / new_diag[..., -1]
    
    for i in range(channels-2, -1, -1):
        x[..., i] = (new_b[..., i] - A_upper[..., i] * x[..., i+1]) / new_diag[..., i]
    
    return x


def _validate_input(t: torch.Tensor, X: torch.Tensor) -> None:
    
    if not t.is_floating_point():
        raise ValueError("t must be floating point.")
    if not X.is_floating_point():
        raise ValueError("X must be floating point.")
    if len(t.shape) != 1:
        raise ValueError(f"t must be one dimensional. It instead has shape {tuple(t.shape)}.")
    
    prev_t_i = -math.inf
    for t_i in t:
        if t_i <= prev_t_i:
            raise ValueError("t must be monotonically increasing.")
        prev_t_i = t_i
    
    if X.ndimension() < 2:
        raise ValueError(f"X must have at least two dimensions (time and channels). It has shape {tuple(X.shape)}.")
    if X.size(-2) != t.size(0):
        raise ValueError("The time dimension of X must equal the length of t.")
    if t.size(0) < 2:
        raise ValueError("Must have a time dimension of size at least 2.")


def _natural_cubic_spline_coeffs_without_missing_values(
    t: torch.Tensor, 
    x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
   
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
        time_diffs_reciprocal = 1.0 / time_diffs
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
        
        knot_derivatives = tridiagonal_solve(
            system_rhs, 
            time_diffs_reciprocal, 
            system_diagonal, 
            time_diffs_reciprocal
        )

        a = x[..., :-1]
        b = knot_derivatives[..., :-1]
        
        two_c = (
            six_path_diffs * time_diffs_reciprocal - 
            4 * knot_derivatives[..., :-1] - 
            2 * knot_derivatives[..., 1:]
        ) * time_diffs_reciprocal
        
        three_d = (
            -six_path_diffs * time_diffs_reciprocal + 
            3 * (knot_derivatives[..., :-1] + knot_derivatives[..., 1:])
        ) * time_diffs_reciprocal_squared

    return a, b, two_c, three_d


def _natural_cubic_spline_coeffs_with_missing_values(
    t: torch.Tensor, 
    x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
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
        
        return (
            cheap_stack(a_pieces, dim=0),
            cheap_stack(b_pieces, dim=0),
            cheap_stack(two_c_pieces, dim=0),
            cheap_stack(three_d_pieces, dim=0)
        )


def _natural_cubic_spline_coeffs_with_missing_values_scalar(
    t: torch.Tensor, 
    x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    not_nan = ~torch.isnan(x)
    path_no_nan = x.masked_select(not_nan)
    
    if path_no_nan.size(0) == 0:
        zeros = torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device)
        return zeros, zeros, zeros, zeros

    need_new_not_nan = False
    x_filled = x.clone()
    
    if torch.isnan(x_filled[0]):
        x_filled[0] = path_no_nan[0]
        need_new_not_nan = True
        
    if torch.isnan(x_filled[-1]):
        x_filled[-1] = path_no_nan[-1]
        need_new_not_nan = True
    
    if need_new_not_nan:
        not_nan = ~torch.isnan(x_filled)
        path_no_nan = x_filled.masked_select(not_nan)
    
    times_no_nan = t.masked_select(not_nan)

    a_pieces_no_nan, b_pieces_no_nan, two_c_pieces_no_nan, three_d_pieces_no_nan = (
        _natural_cubic_spline_coeffs_without_missing_values(times_no_nan, path_no_nan)
    )

    a_pieces, b_pieces, two_c_pieces, three_d_pieces = [], [], [], []
    iter_times_no_nan = iter(times_no_nan)
    iter_coeffs_no_nan = iter(zip(
        a_pieces_no_nan, b_pieces_no_nan, two_c_pieces_no_nan, three_d_pieces_no_nan
    ))
    
    next_time_no_nan = next(iter_times_no_nan)
    prev_time_no_nan = next_time_no_nan
    next_a_no_nan, next_b_no_nan, next_two_c_no_nan, next_three_d_no_nan = next(iter_coeffs_no_nan)
    
    for time in t[:-1]:
        while time >= next_time_no_nan:
            prev_time_no_nan = next_time_no_nan
            try:
                next_time_no_nan = next(iter_times_no_nan)
                next_a_no_nan, next_b_no_nan, next_two_c_no_nan, next_three_d_no_nan = next(iter_coeffs_no_nan)
            except StopIteration:
                break
                
        offset = prev_time_no_nan - time
        a_inner = (0.5 * next_two_c_no_nan - next_three_d_no_nan * offset / 3) * offset
        a_pieces.append(next_a_no_nan + (a_inner - next_b_no_nan) * offset)
        b_pieces.append(next_b_no_nan + (next_three_d_no_nan * offset - next_two_c_no_nan) * offset)
        two_c_pieces.append(next_two_c_no_nan - 2 * next_three_d_no_nan * offset)
        three_d_pieces.append(next_three_d_no_nan)

    return (
        cheap_stack(a_pieces, dim=0),
        cheap_stack(b_pieces, dim=0),
        cheap_stack(two_c_pieces, dim=0),
        cheap_stack(three_d_pieces, dim=0)
    )


def natural_cubic_spline_coeffs(
    t: torch.Tensor, 
    x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    _validate_input(t, x)
    
    x_transposed = x.transpose(-1, -2)
    
    if torch.isnan(x).any():
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, x_transposed)
    else:
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_without_missing_values(t, x_transposed)
    
    a = a.transpose(-1, -2)
    b = b.transpose(-1, -2)
    c = two_c.transpose(-1, -2) / 2
    d = three_d.transpose(-1, -2) / 3
    
    return t, a, b, c, d


class NaturalCubicSpline:
    
    def __init__(self, coeffs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        
        t, a, b, c, d = coeffs
        self._t = t
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._length = b.size(-2)

    def _interpret_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       
        index = torch.bucketize(t.detach(), self._t) - 1
        index = index.clamp(0, self._length - 1)
        
        fractional_part = t - self._t[index]
        
        return fractional_part, index

    def evaluate(self, t: torch.Tensor) -> torch.Tensor:
        
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        
        result = self._d[..., index, :]
        result = result * fractional_part + self._c[..., index, :]
        result = result * fractional_part + self._b[..., index, :]
        result = result * fractional_part + self._a[..., index, :]
        
        return result

    def derivative(self, t: torch.Tensor, order: int = 1) -> torch.Tensor:
        
        if order not in (1, 2):
            raise ValueError("Derivative is only implemented for orders 1 and 2.")
            
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        
        if order == 1:
            result = 3 * self._d[..., index, :]
            result = result * fractional_part + 2 * self._c[..., index, :]
            result = result * fractional_part + self._b[..., index, :]
        else:  
            result = 6 * self._d[..., index, :]
            result = result * fractional_part + 2 * self._c[..., index, :]
            
        return result




@dataclass
class ModelResult:
    name: str
    values: np.ndarray
    color: str
    dash: str = 'solid'
    uncertainty: Optional[np.ndarray] = None


class YieldCurveSplineFitter:
    
    TREASURY_SERIES = {
        "DGS1MO": 1/12,  # 1 month
        "DGS3MO": 0.25,  # 3 months
        "DGS6MO": 0.5,   # 6 months
        "DGS1": 1,       # 1 year
        "DGS2": 2,       # 2 years
        "DGS3": 3,       # 3 years
        "DGS5": 5,       # 5 years
        "DGS7": 7,       # 7 years
        "DGS10": 10,     # 10 years
        "DGS20": 20,     # 20 years
        "DGS30": 30      # 30 years
    }
    
    def __init__(
        self, 
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        use_simulated_data: bool = False
    ):
        
        self.start_date = start_date or (datetime.date.today() - datetime.timedelta(days=60))
        self.end_date = end_date or datetime.date.today()
        self.use_simulated_data = use_simulated_data
        
        self.data = None
        self.maturities = None
        self.yields = None
        
        self.spline = None
        self.gpr_model = None
        self.mlp_model = None
        self.models_fitted = set()
        
        self.gpr_scaler_X = None
        self.gpr_scaler_y = None
        self.mlp_scaler_X = None
        self.mlp_scaler_y = None
        
        logger.info(f"Initialized YieldCurveSplineFitter for period {self.start_date} to {self.end_date}")

    @log_execution
    def fetch_yield_curve(self) -> pd.DataFrame:
        
        fetched_data = {}
        
        if self.use_simulated_data or not HAS_DATAREADER:
            logger.info("Using simulated yield curve data")
            fetched_data = {
                1/12: 0.05,  # 1 month
                0.25: 0.08,  # 3 months
                0.5: 0.10,   # 6 months
                1: 0.15,     # 1 year
                2: 0.30,     # 2 years
                3: 0.40,     # 3 years
                5: 0.70,     # 5 years
                7: 0.85,     # 7 years
                10: 1.00,    # 10 years
                20: 1.50,    # 20 years
                30: 1.75     # 30 years
            }
        else:
            for code, maturity in self.TREASURY_SERIES.items():
                try:
                    df = web.DataReader(code, "fred", self.start_date, self.end_date)
                    df.dropna(inplace=True)
                    
                    if not df.empty:
                        value = float(df.iloc[-1, 0])
                        fetched_data[maturity] = value
                        logger.info(f"Fetched {code}: {value:.2f}% for maturity {maturity} years")
                    else:
                        logger.warning(f"Empty data for {code}, skipping series")
                except Exception as e:
                    logger.warning(f"Could not fetch data for {code}: {str(e)} - Skipping series")

            if not fetched_data:
                logger.warning("No yield data could be fetched. Using simulated data.")
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
        
        logger.info(f"Yield curve data prepared with {len(self.data)} points")
        return self.data

    @log_execution
    def fit_spline(self) -> NaturalCubicSpline:
        
        self._ensure_data_available()
        
        t = torch.tensor(self.data["Maturity"].values, dtype=torch.float64)
        x = torch.tensor(self.data["Yield"].values, dtype=torch.float64).unsqueeze(-1)
        
        coeffs = natural_cubic_spline_coeffs(t, x)
        self.spline = NaturalCubicSpline(coeffs)
        self.models_fitted.add('spline')
        
        logger.info("Cubic spline fitting completed")
        return self.spline

    @log_execution
    def fit_gpr(self) -> Optional[GaussianProcessRegressor]:
        
        self._ensure_data_available()
        
        if not HAS_SKLEARN:
            logger.warning("Cannot fit GPR model: scikit-learn not available")
            return None
        
        X = self.maturities.reshape(-1, 1)
        y = self.yields
        
        self.gpr_scaler_X = StandardScaler().fit(X)
        self.gpr_scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        
        X_scaled = self.gpr_scaler_X.transform(X)
        y_scaled = self.gpr_scaler_y.transform(y.reshape(-1, 1)).ravel()
        
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) * 
            Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
            WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
        )
        
        gpr = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10, 
            random_state=42,
            normalize_y=False 
        )
        gpr.fit(X_scaled, y_scaled)
        
        self.gpr_model = gpr
        self.models_fitted.add('gpr')
        
        logger.info(f"GPR fitting completed. Learned kernel: {gpr.kernel_}")
        return gpr

    @log_execution
    def fit_mlp(self) -> Optional[MLPRegressor]:
        
        self._ensure_data_available()
        
        if not HAS_SKLEARN:
            logger.warning("Cannot fit MLP model: scikit-learn not available")
            return None
        
        X = self.maturities.reshape(-1, 1)
        y = self.yields
        
        self.mlp_scaler_X = StandardScaler().fit(X)
        self.mlp_scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        
        X_scaled = self.mlp_scaler_X.transform(X)
        y_scaled = self.mlp_scaler_y.transform(y.reshape(-1, 1)).ravel()
        
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=10000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        
        mlp.fit(X_scaled, y_scaled)
        
        self.mlp_model = mlp
        self.models_fitted.add('mlp')
        
        logger.info("MLP fitting completed")
        return mlp

    def _ensure_data_available(self) -> None:
        
        if self.data is None:
            logger.info("Yield curve data not available. Fetching automatically.")
            self.fetch_yield_curve()
            
        if self.data is None or self.data.empty:
            raise RuntimeError("Failed to fetch yield curve data")

    def predict(self, maturities: Union[float, List[float], np.ndarray]) -> Dict[str, np.ndarray]:
        
        if not self.models_fitted:
            raise RuntimeError("No models have been fitted. Call fit_spline(), fit_gpr(), or fit_mlp() first.")
            
        if isinstance(maturities, (float, int)):
            maturities = np.array([float(maturities)])
        elif isinstance(maturities, list):
            maturities = np.array(maturities)
            
        results = {}
        
        if 'spline' in self.models_fitted:
            t_pred = torch.tensor(maturities, dtype=torch.float64)
            spline_pred = self.spline.evaluate(t_pred).squeeze().detach().numpy()
            results['spline'] = spline_pred
            
        if 'gpr' in self.models_fitted:
            X_pred = maturities.reshape(-1, 1)
            X_pred_scaled = self.gpr_scaler_X.transform(X_pred)
            y_pred_scaled, sigma = self.gpr_model.predict(X_pred_scaled, return_std=True)
            y_pred = self.gpr_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            uncertainty = sigma * self.gpr_scaler_y.scale_[0]
            results['gpr'] = y_pred
            results['gpr_uncertainty'] = uncertainty
            
        if 'mlp' in self.models_fitted:
            X_pred = maturities.reshape(-1, 1)
            X_pred_scaled = self.mlp_scaler_X.transform(X_pred)
            y_pred_scaled = self.mlp_model.predict(X_pred_scaled)
            y_pred = self.mlp_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            results['mlp'] = y_pred
            
        return results
    
    @log_execution
    def plot(self, point_density: int = 200, show_uncertainty: bool = True) -> go.Figure:
       
        self._ensure_data_available()
        
        if not self.models_fitted:
            logger.warning("No models have been fitted. Plot will show only data points.")
        
        t_min, t_max = self.maturities[0], self.maturities[-1]
        t_fine = np.linspace(t_min, t_max, point_density)
        
        model_results = []
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.maturities, 
            y=self.yields, 
            mode='markers', 
            name='Yield Curve Data',
            marker=dict(size=10, color='red')
        ))
        
        if self.models_fitted:
            predictions = self.predict(t_fine)
            
            if 'spline' in predictions:
                fig.add_trace(go.Scatter(
                    x=t_fine, 
                    y=predictions['spline'], 
                    mode='lines', 
                    name='Cubic Spline',
                    line=dict(color='blue', width=2)
                ))
            
            if 'gpr' in predictions:
                fig.add_trace(go.Scatter(
                    x=t_fine, 
                    y=predictions['gpr'],
                    mode='lines', 
                    name='GPR Interpolation',
                    line=dict(color='green', dash='dash', width=2)
                ))
                
                if show_uncertainty and 'gpr_uncertainty' in predictions:
                    upper = predictions['gpr'] + 1.96 * predictions['gpr_uncertainty']
                    lower = predictions['gpr'] - 1.96 * predictions['gpr_uncertainty']
                    
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([t_fine, t_fine[::-1]]),
                        y=np.concatenate([upper, lower[::-1]]),
                        fill='toself',
                        fillcolor='rgba(0, 255, 0, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name='GPR 95% Confidence'
                    ))
            
            if 'mlp' in predictions:
                fig.add_trace(go.Scatter(
                    x=t_fine, 
                    y=predictions['mlp'],
                    mode='lines', 
                    name='MLP Interpolation',
                    line=dict(color='purple', dash='dot', width=2)
                ))
        
        fig.update_layout(
            title={
                'text': 'U.S. Treasury Yield Curve Interpolation',
                'font': {'size': 24}
            },
            xaxis_title={
                'text': 'Maturity (years)',
                'font': {'size': 18}
            },
            yaxis_title={
                'text': 'Yield (%)',
                'font': {'size': 18}
            },
            legend={'font': {'size': 16}},
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        
        return fig
    
    @log_execution
    def plot_derivatives(self, order: int = 1, point_density: int = 200) -> go.Figure:
        
        if 'spline' not in self.models_fitted:
            raise RuntimeError("Spline not fitted. Call fit_spline() first.")
            
        if order not in (1, 2):
            raise ValueError("Derivative order must be 1 or 2.")
        
        t_min, t_max = self.maturities[0], self.maturities[-1]
        t_fine = torch.linspace(t_min, t_max, point_density, dtype=torch.float64)
        
        deriv_vals = self.spline.derivative(t_fine, order=order).squeeze().detach().numpy()
        
        fig = go.Figure()
        
        derivative_name = "First Derivative" if order == 1 else "Second Derivative"
        y_axis_title = "Rate of Change (%/year)" if order == 1 else "Curvature (%/year²)"
        
        fig.add_trace(go.Scatter(
            x=t_fine.detach().numpy(), 
            y=deriv_vals, 
            mode='lines', 
            name=derivative_name,
            line=dict(color='blue', width=2)
        ))
        
        fig.add_shape(
            type="line",
            x0=t_min,
            y0=0,
            x1=t_max,
            y1=0,
            line=dict(color="black", dash="dash", width=1)
        )
        
        fig.update_layout(
            title={
                'text': f'Yield Curve {derivative_name}',
                'font': {'size': 24}
            },
            xaxis_title={
                'text': 'Maturity (years)',
                'font': {'size': 18}
            },
            yaxis_title={
                'text': y_axis_title,
                'font': {'size': 18}
            },
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def compare_models(self, maturities: List[float] = None) -> pd.DataFrame:
        
        if not self.models_fitted:
            raise RuntimeError("No models have been fitted.")
            
        eval_maturities = maturities if maturities else self.maturities
        
        predictions = self.predict(eval_maturities)
        
        df = pd.DataFrame({"Maturity": eval_maturities})
        
        if maturities is None:
            df["Actual"] = self.yields
            
        for model_name, pred in predictions.items():
            if not model_name.endswith('_uncertainty'):
                df[model_name.capitalize()] = pred
                
        return df
    
    def summarize_models(self) -> Dict[str, Dict[str, Any]]:
        
        summary = {}
        
        if 'spline' in self.models_fitted:
            summary['spline'] = {
                'type': 'Natural Cubic Spline',
                'fitted': True
            }
            
        if 'gpr' in self.models_fitted:
            summary['gpr'] = {
                'type': 'Gaussian Process Regression',
                'kernel': str(self.gpr_model.kernel_),
                'fitted': True
            }
            
        if 'mlp' in self.models_fitted:
            summary['mlp'] = {
                'type': 'Multi-Layer Perceptron',
                'architecture': f"Input(1) -> Hidden{self.mlp_model.hidden_layer_sizes} -> Output(1)",
                'activation': self.mlp_model.activation,
                'solver': self.mlp_model.solver,
                'fitted': True,
                'iterations': self.mlp_model.n_iter_
            }
            
        return summary
    
    def fit_all_models(self) -> Dict[str, bool]:
        
        results = {}
        
        try:
            self.fit_spline()
            results['spline'] = True
        except Exception as e:
            logger.error(f"Error fitting spline: {str(e)}")
            results['spline'] = False
            
        if HAS_SKLEARN:
            try:
                self.fit_gpr()
                results['gpr'] = True
            except Exception as e:
                logger.error(f"Error fitting GPR: {str(e)}")
                results['gpr'] = False
        else:
            results['gpr'] = False
            
        if HAS_SKLEARN:
            try:
                self.fit_mlp()
                results['mlp'] = True
            except Exception as e:
                logger.error(f"Error fitting MLP: {str(e)}")
                results['mlp'] = False
        else:
            results['mlp'] = False
            
        return results


@log_execution
def analyze_yield_curve(
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    use_simulated_data: bool = False,
    fit_all: bool = True,
    plot_results: bool = True
) -> Tuple[YieldCurveSplineFitter, Optional[go.Figure]]:
    
    fitter = YieldCurveSplineFitter(
        start_date=start_date,
        end_date=end_date,
        use_simulated_data=use_simulated_data
    )
    
    fitter.fetch_yield_curve()
    
    if fit_all:
        fitter.fit_all_models()
    else:
        fitter.fit_spline()
    
    fig = None
    if plot_results:
        fig = fitter.plot()
    
    return fitter, fig


def main():
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze U.S. Treasury yield curve.')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--simulated', action='store_true', help='Use simulated data')
    parser.add_argument('--no-plot', action='store_true', help='Do not generate plot')
    parser.add_argument('--spline-only', action='store_true', help='Fit only the spline model')
    
    args = parser.parse_args()
    
    start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else None
    end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else None
    
    fitter, fig = analyze_yield_curve(
        start_date=start_date,
        end_date=end_date,
        use_simulated_data=args.simulated,
        fit_all=not args.spline_only,
        plot_results=not args.no_plot
    )
    
    if fig:
        fig.show()
    
    return 0


if __name__ == "__main__":
    main()
