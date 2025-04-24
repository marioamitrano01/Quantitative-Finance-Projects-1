import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

import sys
import logging
import functools
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_execution(func):
    """Decorator to log execution time of methods."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Finished {func.__name__} in {end_time - start_time:.2f}s")
        return result
    return wrapper


class ETHUSDTForecast:
    def __init__(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "1d",
        start_date: pd.Timestamp = pd.Timestamp("2020-01-01", tz="UTC"),
        forecast_horizon: int = 30,
        lstm_epochs: int = 50,
        lstm_batch_size: int = 16,
        seq_length: int = 20,
        weight_garch: float = 0.5,
        weight_lstm: float = 0.5,
    ):
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.forecast_horizon = forecast_horizon
        self.lstm_epochs = lstm_epochs
        self.lstm_batch_size = lstm_batch_size
        self.seq_length = seq_length
        self.weight_garch = weight_garch
        self.weight_lstm = weight_lstm

        self.df_eth = None
        self.rstock = None
        self.garch_model = None
        self.garch_res = None
        self.std_resid = None

    @log_execution
    def fetch_data(self, end_date: pd.Timestamp = None):
        """Fetch OHLCV data from Binance with pagination."""
        if end_date is None:
            end_date = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)

        start_ts = int(self.start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        all_candles = []
        base_url = "https://api.binance.com"
        endpoint = "/api/v3/klines"

        while True:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "startTime": start_ts,
                "limit": 1000
            }
            resp = requests.get(base_url + endpoint, params=params)
            data = resp.json()
            if not isinstance(data, list) or len(data) == 0:
                break
            all_candles.extend(data)
            last_candle_close_time = data[-1][6]
            if last_candle_close_time >= end_ts:
                break
            start_ts = last_candle_close_time + 1

        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ]
        df = pd.DataFrame(all_candles, columns=columns)
        numeric_cols = ["open", "high", "low", "close", "volume",
                        "quote_asset_volume", "num_trades",
                        "taker_buy_base_vol", "taker_buy_quote_vol"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms', utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms', utc=True)
        df.set_index("open_time", inplace=True)
        df.sort_index(inplace=True)
        df = df[df.index <= end_date]
        self.df_eth = df

        logger.info(f"Fetched {len(df)} daily candles for {self.symbol} from {self.start_date.date()} to {end_date.date()}.")

    @log_execution
    def compute_returns(self):
        """Compute log-returns from closing prices."""
        if self.df_eth is None:
            logger.error("Dataframe is empty. Fetch data first.")
            sys.exit(1)
        self.df_eth["return"] = np.log(self.df_eth["close"]).diff()
        self.rstock = self.df_eth["return"].dropna()
        if len(self.rstock) < 10:
            logger.error("Not enough return data. Exiting.")
            sys.exit(1)

    @log_execution
    def perform_diagnostics(self):
        """Run Ljung-Box and ARCH LM tests on returns."""
        logger.info("--- Ljung-Box Test on Returns (Lag=20) ---")
        lb_test = sm.stats.diagnostic.acorr_ljungbox(self.rstock, lags=[20], return_df=True)
        print(lb_test)

        logger.info("--- ARCH LM Test on Returns ---")
        arch_test = sm.stats.diagnostic.het_arch(self.rstock)
        print(f"LM stat = {arch_test[0]:.3f}, LM p-value = {arch_test[1]:.3f}")
        print(f"F stat  = {arch_test[2]:.3f}, F p-value  = {arch_test[3]:.3f}")

    @log_execution
    def fit_garch_model(self):
        """Fit an AR(1)-GARCH(1,1) model and compute standardized residuals."""
        self.garch_model = arch_model(
            self.rstock,
            mean='AR', lags=1,
            vol='Garch', p=1, q=1,
            dist='normal',
            rescale=False
        )
        self.garch_res = self.garch_model.fit(update_freq=5, disp='off')
        print("\n--- AR(1)-GARCH(1,1) Model Summary ---")
        print(self.garch_res.summary())
        self.std_resid = self.garch_res.resid / self.garch_res.conditional_volatility

    @log_execution
    def plot_residual_diagnostics(self):
        """Produce a Q-Q plot and histogram of standardized residuals using Plotly."""
        res = self.std_resid.copy()
        res = res[~np.isnan(res)]
        res_sorted = np.sort(res)
        n = len(res_sorted)
        qq_theoretical = scs.norm.ppf((np.arange(n) + 0.5) / n)
        slope, intercept, _, _, _ = scs.linregress(qq_theoretical, res_sorted)
        line_qq = intercept + slope * qq_theoretical

        x_min, x_max = res.min(), res.max()
        x_vals = np.linspace(x_min, x_max, 200)
        y_vals = scs.norm.pdf(x_vals, loc=0, scale=1)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Q-Q Plot of Std. Residuals", "Histogram of Std. Residuals")
        )
        fig.add_trace(
            go.Scatter(
                x=qq_theoretical,
                y=res_sorted,
                mode='markers',
                name='QQ Points'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=qq_theoretical,
                y=line_qq,
                mode='lines',
                line=dict(color='red'),
                name='Best-Fit Line'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(
                x=res,
                histnorm='probability density',
                marker_color='skyblue',
                name='Residuals Hist'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Normal PDF'
            ),
            row=1, col=2
        )
        fig.update_layout(
            height=500,
            width=1000,
            showlegend=True,
            title="Diagnostic Plots: Standardized Residuals"
        )
        fig.update_xaxes(title="Theoretical Quantiles", row=1, col=1)
        fig.update_yaxes(title="Empirical Quantiles", row=1, col=1)
        fig.update_xaxes(title="Residual Value", row=1, col=2)
        fig.update_yaxes(title="Probability Density", row=1, col=2)
        fig.show()

    @log_execution
    def forecast_garch(self):
        """Forecast future mean and volatility using the fitted GARCH model."""
        forecast = self.garch_res.forecast(horizon=self.forecast_horizon, reindex=False)
        fc_mean = forecast.mean.values[-1]       # AR(1) predicted mean log-returns
        fc_var  = forecast.variance.values[-1]    # GARCH predicted variance
        fc_vol  = np.sqrt(fc_var)
        return fc_mean, fc_var, fc_vol

    @staticmethod
    def create_sequences(data, seq_length):
        """Prepare a time series for LSTM input."""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    @log_execution
    def forecast_lstm_volatility(self, hist_vol, fc_horizon, seq_length, lstm_epochs, lstm_batch_size):
        """Train an LSTM on historical volatility and produce a forecast."""
        vol_data = hist_vol.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        vol_scaled = scaler.fit_transform(vol_data)
        X, y = self.create_sequences(vol_scaled, seq_length)

        logger.info(f"Training LSTM on {len(X)} sequences (epochs={lstm_epochs}, batch_size={lstm_batch_size})...")
        model_lstm = Sequential()
        model_lstm.add(Input(shape=(seq_length, 1)))
        model_lstm.add(LSTM(50, activation='tanh', return_sequences=False))
        model_lstm.add(Dropout(0.2))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mse')
        model_lstm.fit(X, y, epochs=lstm_epochs, batch_size=lstm_batch_size, verbose=0)

        lstm_forecast_scaled = []
        current_seq = vol_scaled[-seq_length:].copy()
        for i in range(fc_horizon):
            pred = model_lstm.predict(current_seq.reshape(1, seq_length, 1), verbose=0)
            lstm_forecast_scaled.append(pred[0, 0])
            current_seq = np.append(current_seq[1:], [[pred[0, 0]]], axis=0)
        lstm_forecast_vol = scaler.inverse_transform(
            np.array(lstm_forecast_scaled).reshape(-1, 1)
        ).flatten()
        return lstm_forecast_vol

    @log_execution
    def run_monte_carlo_simulation(self, last_price, fc_mean, combined_vol, fc_horizon, num_sims=200):
        """Perform Monte Carlo simulation for future price paths."""
        rng = np.random.default_rng(seed=42)
        mc_paths = np.zeros((fc_horizon, num_sims))
        for sim in range(num_sims):
            sim_price = last_price
            for i in range(fc_horizon):
                epsilon = rng.normal(0, 1)
                daily_return = fc_mean[i] + combined_vol[i] * epsilon
                sim_price *= np.exp(daily_return)
                mc_paths[i, sim] = sim_price
        p10 = np.percentile(mc_paths, 10, axis=1)
        p50 = np.percentile(mc_paths, 50, axis=1)
        p90 = np.percentile(mc_paths, 90, axis=1)
        return p10, p50, p90, mc_paths

    @log_execution
    def plot_forecasts(self, fc_mean, fc_vol, lstm_vol, combined_vol, point_forecast_prices, future_dates, p10, p50, p90, hist_vol):
        """Plot historical prices, deterministic and Monte Carlo forecasts along with volatility forecasts."""
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                "ETH/USDT Price: Historical, Deterministic & Monte Carlo Forecast",
                "GARCH & LSTM Volatility Forecasts (Historical + Future)"
            )
        )
        # Historical Price
        fig.add_trace(
            go.Scatter(
                x=self.df_eth.index,
                y=self.df_eth["close"],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        # Deterministic Point Forecast
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=point_forecast_prices,
                mode='lines+markers',
                name='Point Forecast Price',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        # Monte Carlo Forecast Bands
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=p90,
                mode='lines',
                line=dict(width=0, color='rgba(255,0,0,0.2)'),
                name='90th Percentile',
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=p10,
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=0, color='rgba(255,0,0,0.2)'),
                name='10–90% MC Band',
                showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=p50,
                mode='lines',
                line=dict(color='magenta', dash='dot'),
                name='Median MC Path'
            ),
            row=1, col=1
        )
        # Volatility Forecasts
        fig.add_trace(
            go.Scatter(
                x=hist_vol.index,
                y=hist_vol,
                mode='lines',
                name='Historical Vol (GARCH)',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=fc_vol,
                mode='lines+markers',
                name='GARCH Forecast Vol',
                line=dict(color='red', dash='dot')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=lstm_vol,
                mode='lines+markers',
                name='LSTM Forecast Vol',
                line=dict(color='orange', dash='dot')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=combined_vol,
                mode='lines+markers',
                name='Combined Forecast Vol',
                line=dict(color='purple', dash='dash')
            ),
            row=2, col=1
        )

        fig.update_layout(
            title="GARCH + LSTM Volatility Forecast on ETH/USDT",
            height=900,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0)"),
            xaxis=dict(title="Date"),
            yaxis=dict(title="ETH Price (USDT)"),
        )
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(title="Volatility (Std. Dev.)", row=2, col=1)
        fig.show()

    @log_execution
    def run_forecast(self):
        """Run the complete forecasting workflow."""
        # Set the end date as yesterday.
        yesterday = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
        self.fetch_data(end_date=yesterday)
        self.compute_returns()
        self.perform_diagnostics()
        self.fit_garch_model()

        logger.info("--- Residual Analysis ---")
        lb_resid = sm.stats.diagnostic.acorr_ljungbox(self.std_resid.dropna(), lags=[20], return_df=True)
        print("\nLjung-Box test on standardized residuals (lag=20):")
        print(lb_resid)
        arch_test_resid = sm.stats.diagnostic.het_arch(self.std_resid.dropna())
        print("\nARCH LM test on standardized residuals:")
        print(f"LM stat = {arch_test_resid[0]:.3f}, LM p-value = {arch_test_resid[1]:.3f}")
        print(f"F stat  = {arch_test_resid[2]:.3f}, F p-value  = {arch_test_resid[3]:.3f}")

        self.plot_residual_diagnostics()

        # GARCH Forecast
        fc_mean, fc_var, fc_vol = self.forecast_garch()
        last_price = self.df_eth["close"].iloc[-1]
        forecast_start_date = self.df_eth.index[-1] + pd.Timedelta(days=1)
        point_forecast_prices = [last_price]
        future_dates = []
        for i in range(self.forecast_horizon):
            future_date = forecast_start_date + pd.Timedelta(days=i)
            future_dates.append(future_date)
            next_price = point_forecast_prices[-1] * np.exp(fc_mean[i])
            point_forecast_prices.append(next_price)
        point_forecast_prices = point_forecast_prices[1:]

        # LSTM Volatility Forecast
        hist_vol = self.garch_res.conditional_volatility.dropna()
        lstm_vol = self.forecast_lstm_volatility(hist_vol, self.forecast_horizon, self.seq_length, self.lstm_epochs, self.lstm_batch_size)
        combined_vol = (self.weight_garch * fc_vol + self.weight_lstm * lstm_vol)

        p10, p50, p90, _ = self.run_monte_carlo_simulation(last_price, fc_mean, combined_vol, self.forecast_horizon)

        self.plot_forecasts(fc_mean, fc_vol, lstm_vol, combined_vol, point_forecast_prices, future_dates, p10, p50, p90, hist_vol)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Mean Log-Return": fc_mean,
            "GARCH Vol": fc_vol,
            "LSTM Vol": lstm_vol,
            "Combined Vol": combined_vol,
            "Point Forecast Price": point_forecast_prices,
            "MC Median Price": p50,
            "MC 10%": p10,
            "MC 90%": p90
        }).set_index("Date")

        print("\n--Forecast Summary (First 10 lines)--")
        print(forecast_df.head(10).round(3))
        print("\nDone.")


if __name__ == "__main__":
    forecast_app = ETHUSDTForecast()
    forecast_app.run_forecast()
