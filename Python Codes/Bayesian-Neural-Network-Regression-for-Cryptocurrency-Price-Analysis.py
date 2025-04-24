import argparse
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
import pyro.infer.autoguide as autoguide
import pyro.optim as optim
import plotly.graph_objects as go
import plotly.io as pio
import requests
from sklearn.linear_model import RidgeCV, HuberRegressor






pio.renderers.default = "browser"
pyro.set_rng_seed(42)
torch.manual_seed(42)


@dataclass




class BayesianNNConfig:
    num_svi_steps: int = 1000000
    learning_rate: float = 0.005
    debug: bool = False
    save_trace: bool = False
    trace_filename: Optional[str] = "bnn_trace.pt"


def complex_ml_initialization(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
   


    X_flat = X.reshape(-1, 1)
    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
    ridge_cv.fit(X_flat, y)
    ridge_intercept = ridge_cv.intercept_
    ridge_slope = ridge_cv.coef_[0]
    huber = HuberRegressor(epsilon=1.35, max_iter=1000)
    huber.fit(X_flat, y)
    huber_intercept = huber.intercept_
    huber_slope = huber.coef_[0]
    ml_beta0 = (ridge_intercept + huber_intercept) / 2.0
    ml_beta1 = (ridge_slope + huber_slope) / 2.0
    return ml_beta0, ml_beta1


class BayesianNeuralNetworkRegression:
   


    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 config: BayesianNNConfig,
                 ml_beta0: float,
                 ml_beta1: float,
                 hidden_size: int = 10) -> None:
        self.config = config
        self.X_np = X
        self.y_np = y
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)
        self.ml_beta0 = ml_beta0
        self.ml_beta1 = ml_beta1
        self.guide = None
        self.elbo_losses: List[float] = []







    def model(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        W1 = pyro.sample(
            "W1", 
            dist.Normal(
                torch.zeros(self.hidden_size, 1, device=self.device, dtype=torch.float32),
                torch.ones(self.hidden_size, 1, device=self.device, dtype=torch.float32)
            ).to_event(2)
        )
        b1 = pyro.sample(
            "b1", 
            dist.Normal(
                torch.zeros(self.hidden_size, device=self.device, dtype=torch.float32),
                torch.ones(self.hidden_size, device=self.device, dtype=torch.float32)
            ).to_event(1)
        )
        W2_mean = (self.ml_beta1 / self.hidden_size) * torch.ones(1, self.hidden_size, device=self.device, dtype=torch.float32)
        W2 = pyro.sample(
            "W2", 
            dist.Normal(
                W2_mean,
                torch.ones(1, self.hidden_size, device=self.device, dtype=torch.float32)
            ).to_event(2)
        )
        b2 = pyro.sample(
            "b2", 
            dist.Normal(
                torch.tensor(self.ml_beta0, device=self.device, dtype=torch.float32),
                torch.tensor(1.0, device=self.device, dtype=torch.float32)
            )
        )
        sigma = pyro.sample(
            "sigma", 
            dist.HalfNormal(torch.tensor(1.0, device=self.device, dtype=torch.float32))
        )
        hidden = torch.relu(x.matmul(W1.t()) + b1)
        y_hat = hidden.matmul(W2.t()).squeeze(-1) + b2
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(y_hat, sigma), obs=y)
        return y_hat





    def run_svi(self) -> None:
        self.guide = autoguide.AutoDiagonalNormal(self.model)
        optimizer = optim.Adam({"lr": self.config.learning_rate})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        logging.info("Starting SVI optimization for Bayesian Neural Network Regression...")
        for step in range(self.config.num_svi_steps):
            loss = svi.step(self.X, self.y)
            self.elbo_losses.append(loss)
            if self.config.debug and step % 1000 == 0:
                logging.info(f"Step {step}: ELBO loss = {loss:.2f}")
        logging.info("SVI optimization complete.")
        if self.config.save_trace and self.config.trace_filename:
            torch.save(self.guide.state_dict(), self.config.trace_filename)
            logging.info(f"Variational guide saved to {self.config.trace_filename}")






    def compute_posterior_predictive(self, X_new: np.ndarray, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        if self.guide is None:
            raise ValueError("No variational guide available. Run SVI inference first.")
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32, device=self.device)
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=["obs"])
        samples = predictive(X_new_tensor)
        y_pred_samples = samples["obs"].detach().cpu().numpy()
        mean_pred = y_pred_samples.mean(axis=0)
        std_pred = y_pred_samples.std(axis=0)
        return {"y_pred": y_pred_samples, "mean": mean_pred, "std": std_pred}




    def summary(self, num_samples: int = 1000) -> Dict[str, Dict[str, float]]:
        if self.guide is None:
            raise ValueError("No variational guide available. Run SVI inference first.")
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
        samples = predictive(self.X)
        summary_dict = {}
        for param, vals in samples.items():
            vals_np = vals.detach().cpu().numpy().flatten()
            mean_val = np.mean(vals_np)
            std_val = np.std(vals_np)
            ci_lower, ci_upper = np.percentile(vals_np, [2.5, 97.5])
            summary_dict[param] = {
                "mean": mean_val,
                "std": std_val,
                "2.5%": ci_lower,
                "97.5%": ci_upper,
            }
            logging.info(f"Parameter {param}: {summary_dict[param]}")
        return summary_dict






def fetch_binance_data(symbol: str, interval: str = '1d', limit: int = 1000, api_key: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    base_url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {}
    if api_key:
        headers["X-MBX-APIKEY"] = api_key
    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    timestamps = np.array([entry[0] for entry in data])
    closes = np.array([float(entry[4]) for entry in data])
    return timestamps, closes



def fetch_aligned_binance_data(symbol1: str, symbol2: str, interval: str = '1d', limit: int = 1000, api_key: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1, p1 = fetch_binance_data(symbol1, interval, limit, api_key)
    t2, p2 = fetch_binance_data(symbol2, interval, limit, api_key)
    common_timestamps = np.intersect1d(t1, t2)
    d1 = {t: p for t, p in zip(t1, p1)}
    d2 = {t: p for t, p in zip(t2, p2)}
    prices1 = np.array([d1[t] for t in common_timestamps])
    prices2 = np.array([d2[t] for t in common_timestamps])
    return common_timestamps, prices1, prices2



def plot_results(X: np.ndarray, y: np.ndarray, predictive_results: Dict[str, np.ndarray]) -> None:
    mean_pred = predictive_results["mean"]
    std_pred = predictive_results["std"]
    lower_bound = mean_pred - 2 * std_pred
    upper_bound = mean_pred + 2 * std_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X.ravel(),
        y=y,
        mode='markers',
        name='Observations',
        marker=dict(color='blue', size=8, opacity=0.7)
    ))
    fig.add_trace(go.Scatter(
        x=X.ravel(),
        y=mean_pred,
        mode='lines',
        name='Posterior Predictive Mean',
        line=dict(color='red', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([X.ravel(), X.ravel()[::-1]]),
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='95% Credible Interval'
    ))
    fig.update_layout(
        title='Bayesian Neural Network Regression: Posterior Predictive Check',
        xaxis_title='Bitcoin Price (USDT)',
        yaxis_title='Ethereum Price (USDT)',
        template='plotly_white'
    )
    fig.show()



def plot_elbo_loss(losses: List[float]) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(losses))),
        y=losses,
        mode='lines',
        name='ELBO Loss'
    ))
    fig.update_layout(
        title='SVI ELBO Loss Curve',
        xaxis_title='SVI Step',
        yaxis_title='ELBO Loss',
        template='plotly_white'
    )
    fig.show()




def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian Neural Network Regression with real Binance data (BTC and ETH)"
    )
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of data points to retrieve (max 1000 per Binance API)")
    parser.add_argument("--num_svi_steps", type=int, default=10000, help="Number of SVI optimization steps")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate for SVI")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed output")
    parser.add_argument("--save_trace", action="store_true", help="Save the variational parameters to a file")
    parser.add_argument("--trace_filename", type=str, default="bnn_trace.pt", help="Filename for saving the variational parameters")
    parser.add_argument("--num_predictive_samples", type=int, default=1000, help="Number of samples for the posterior predictive distribution")
    parser.add_argument("--plotly_renderer", type=str, default="browser", help="Plotly renderer (e.g., 'browser', 'notebook')")
    parser.add_argument("--binance_api_key", type=str, default=None, help="Binance API Key (if required)")
    return parser.parse_args()




def main():
    args = parse_arguments()
    pio.renderers.default = args.plotly_renderer
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Fetching real Binance data for BTC and ETH...")
    limit = min(args.n_samples, 1000)
    timestamps, btc_prices, eth_prices = fetch_aligned_binance_data("BTCUSDT", "ETHUSDT", interval='1d', limit=limit, api_key=args.binance_api_key)
    logging.info(f"Number of aligned data points: {len(timestamps)}")
    X = btc_prices.reshape(-1, 1)
    y = eth_prices
    logging.info("Data fetched. Running ML initialization...")
    ml_beta0, ml_beta1 = complex_ml_initialization(X, y)
    logging.info(f"ML Estimates: Intercept (β₀) = {ml_beta0:.3f}, Slope (β₁) = {ml_beta1:.3f}")
    config = BayesianNNConfig(
        num_svi_steps=args.num_svi_steps,
        learning_rate=args.learning_rate,
        debug=args.debug,
        save_trace=args.save_trace,
        trace_filename=args.trace_filename
    )
    bnn = BayesianNeuralNetworkRegression(X, y, config, ml_beta0, ml_beta1, hidden_size=10)
    logging.info("Running SVI for Bayesian Neural Network Regression...")
    bnn.run_svi()
    bnn.summary(num_samples=args.num_predictive_samples)
    logging.info("Computing posterior predictive distribution...")
    predictive_results = bnn.compute_posterior_predictive(X, num_samples=args.num_predictive_samples)
    logging.info("Plotting results...")
    plot_results(X, y, predictive_results)
    if args.debug:
        logging.info("Plotting ELBO loss curve...")
        plot_elbo_loss(bnn.elbo_losses)





if __name__ == "__main__":
    main()
