import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from binance.client import Client 
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, LSTM 
from tensorflow.keras.optimizers import Adam 



def log_function_call(func):
    
    def wrapper(*args, **kwargs):
        print(f"[*] Calling function: {func.__name__}")
        print(f"    - args: {args[1:] if len(args) > 1 else None}")
        print(f"    - kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"[*] Execution completed: {func.__name__}\n")
        return result
    return wrapper



class Config:
    


#insert your keys
    API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY')
    API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_BINANCE_API_SECRET')
    

    SYMBOLS = {
        'bitcoin': 'BTCUSDT',
        'ethereum': 'ETHUSDT',
        'ripple': 'XRPUSDT'
    }
    
    INTERVAL = Client.KLINE_INTERVAL_1DAY 
    START_DATE = "1 Jan 2018"              



    @classmethod
    def get_binance_client(cls):
        return Client(cls.API_KEY, cls.API_SECRET)



class DataLoader:
    

    def __init__(self):
        self.client = Config.get_binance_client()
    


    @log_function_call
    def load_data(self, symbol: str) -> pd.DataFrame:
        

        klines = self.client.get_historical_klines(symbol, Config.INTERVAL, Config.START_DATE)
        df = pd.DataFrame(klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close'] = df['Close'].astype(float)
        df = df[['Date', 'Close']]
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    


    @log_function_call
    def load_all(self) -> dict:
        
        data = {}
        for crypto, symbol in Config.SYMBOLS.items():
            data[crypto] = self.load_data(symbol)
        return data



class ReturnsCalculator:
    


    def __init__(self, window_daily=20, window_weekly=8, window_monthly=6):
        self.window_daily = window_daily
        self.window_weekly = window_weekly
        self.window_monthly = window_monthly
    


    @staticmethod
    def compute_log_return(df, shift_days: int):
        return np.log(df["Close"] / df["Close"].shift(shift_days))
    


    @log_function_call
    def add_returns_and_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log_return_1d"]  = self.compute_log_return(df, 1)
        df["log_return_7d"]  = self.compute_log_return(df, 7)
        df["log_return_30d"] = self.compute_log_return(df, 30)
        df = df.dropna().reset_index(drop=True)
        
        df["vol_daily"]   = df["log_return_1d"].rolling(window=self.window_daily).std()
        df["vol_weekly"]  = df["log_return_7d"].rolling(window=self.window_weekly).std()
        df["vol_monthly"] = df["log_return_30d"].rolling(window=self.window_monthly).std()
        df = df.dropna().reset_index(drop=True)
        return df



class PortfolioOptimizer:
   


    def __init__(self, num_portfolios=100000, risk_aversion=1.0):
        self.num_portfolios = num_portfolios
        self.risk_aversion = risk_aversion
    


    @log_function_call
    def optimize(self, combined_df: pd.DataFrame):
        returns = np.log(combined_df / combined_df.shift(1)).dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(mean_returns)
        

        results = np.zeros((3, self.num_portfolios))
        weights_record = []
        
        for i in range(self.num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            objective = portfolio_return - self.risk_aversion * portfolio_std
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std
            results[2, i] = objective
        
        
        max_objective_idx = np.argmax(results[2])
        optimal_weights = weights_record[max_objective_idx]
        portfolios_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Objective"])
        for i, crypto in enumerate(combined_df.columns):
            portfolios_df[crypto + "_weight"] = [w[i] for w in weights_record]
        
        return optimal_weights, portfolios_df
    


    @log_function_call


    def plot_efficient_frontier(self, portfolios_df: pd.DataFrame, ml_portfolio_point: dict,
                                  mc_optimal_weights: np.array, ml_optimal_weights: np.array,
                                  asset_names: list):
        

        def format_weights(weights):
            return "<br>".join([f"{name.upper()}: {w:.2%}" for name, w in zip(asset_names, weights)])
        
        mc_annotation = format_weights(mc_optimal_weights)
        ml_annotation = format_weights(ml_optimal_weights)
        
        mc_scatter = go.Scatter3d(
            x=portfolios_df["Volatility"],
            y=portfolios_df["Return"],
            z=portfolios_df["Objective"],
            mode="markers",
            marker=dict(size=3, color=portfolios_df["Objective"], colorscale="Viridis", opacity=0.8),
            name="Monte Carlo Portfolios",
            hovertemplate="Volatility: %{x:.4f}<br>Return: %{y:.4f}<br>Objective: %{z:.4f}<extra></extra>"
        )
        
        optimal_mc = portfolios_df.loc[portfolios_df["Objective"].idxmax()]
        mc_opt_scatter = go.Scatter3d(
            x=[optimal_mc["Volatility"]],
            y=[optimal_mc["Return"]],
            z=[optimal_mc["Objective"]],
            mode="markers+text",
            marker=dict(color="red", size=8),
            text=[mc_annotation],
            textposition="top center",
            name="MC Optimal Portfolio",
            hovertemplate="MC Optimal<br>Volatility: %{x:.4f}<br>Return: %{y:.4f}<br>Objective: %{z:.4f}<extra></extra>"
        )
        


        ml_opt_scatter = go.Scatter3d(
            x=[ml_portfolio_point['Volatility']],
            y=[ml_portfolio_point['Return']],
            z=[ml_portfolio_point['Objective']],
            mode="markers+text",
            marker=dict(color="orange", size=8),
            text=[ml_annotation],
            textposition="top center",
            name="ML Optimized Portfolio",
            hovertemplate="ML Optimal<br>Volatility: %{x:.4f}<br>Return: %{y:.4f}<br>Objective: %{z:.4f}<extra></extra>"
        )
        
        fig = go.Figure(data=[mc_scatter, mc_opt_scatter, ml_opt_scatter])
        fig.update_layout(
            title="3D Efficient Frontier with Portfolio Allocations",
            scene=dict(
                xaxis=dict(title="Volatility (Std. Deviation)", gridcolor="gray", backgroundcolor="black"),
                yaxis=dict(title="Expected Return", gridcolor="gray", backgroundcolor="black"),
                zaxis=dict(title="Objective", gridcolor="gray", backgroundcolor="black")
            ),
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white")
        )
        fig.show()



class MLPortfolioOptimizer:
   
    def __init__(self, window_size=30, epochs=200, batch_size=32, learning_rate=0.001, risk_aversion=1.0):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.risk_aversion = risk_aversion
        self.model = None



    def build_model(self, num_assets):
        model = Sequential()
        model.add(LSTM(64, return_sequences=False, input_shape=(self.window_size, num_assets)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_assets, activation='softmax'))
        optimizer = Adam(learning_rate=self.learning_rate)
        


        def objective_loss(y_true, y_pred):
            portfolio_returns = tf.reduce_sum(y_true * y_pred, axis=1)
            mean_ret = tf.reduce_mean(portfolio_returns)
            std_ret = tf.math.reduce_std(portfolio_returns)
            objective = mean_ret - self.risk_aversion * std_ret
            epsilon = 1e-6
            entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + epsilon), axis=1)
            lambda_entropy = 0.1 
            return - (objective + lambda_entropy * tf.reduce_mean(entropy))
        
        model.compile(optimizer=optimizer, loss=objective_loss)
        self.model = model



    def prepare_data(self, combined_df):
       
        returns_df = np.log(combined_df / combined_df.shift(1)).dropna()
        data = returns_df.values  # shape (T, num_assets)
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i])
            y.append(data[i])
        X = np.array(X)
        y = np.array(y)
        return X, y



    @log_function_call
    def train(self, X, y):
        num_assets = X.shape[2]
        if self.model is None:
            self.build_model(num_assets)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)



    @log_function_call
    def optimize(self, X):
       
        weights_pred = self.model.predict(X)
        optimal_weights = np.mean(weights_pred, axis=0)
        return optimal_weights



class CryptoPortfolioApplication:
    


    def __init__(self):
        self.data_loader = DataLoader()
        self.portfolio_optimizer = PortfolioOptimizer(num_portfolios=5000, risk_aversion=1.0)
        self.ml_optimizer = MLPortfolioOptimizer(window_size=30, epochs=200, batch_size=32, learning_rate=0.001, risk_aversion=1.0)
        self.combined_df = None
    


    @log_function_call
    def run(self):
        data = self.data_loader.load_all()
        
        combined_df = None
        for crypto in data:
            df = data[crypto][["Date", "Close"]].set_index("Date")
            if combined_df is None:
                combined_df = df.rename(columns={"Close": crypto})
            else:
                combined_df = combined_df.join(df.rename(columns={"Close": crypto}), how="inner")
        combined_df = combined_df.dropna()
        self.combined_df = combined_df
        


        optimal_mc_weights, portfolios_df = self.portfolio_optimizer.optimize(combined_df)
        print("Optimal Portfolio Weights (Monte Carlo Simulation):")
        for crypto, weight in zip(combined_df.columns, optimal_mc_weights):
            print(f"  {crypto.upper()}: {weight:.2%}")
        
        X, y = self.ml_optimizer.prepare_data(combined_df)
        split_idx = int(0.8 * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        self.ml_optimizer.train(X_train, y_train)
        
        ml_optimal_weights = self.ml_optimizer.optimize(X_test)
        print("\nOptimal Portfolio Weights (ML Optimization):")
        for crypto, weight in zip(combined_df.columns, ml_optimal_weights):
            print(f"  {crypto.upper()}: {weight:.2%}")
        
        returns = np.log(combined_df / combined_df.shift(1)).dropna()
        ml_portfolio_return = np.sum(returns.mean() * ml_optimal_weights)
        cov_matrix = returns.cov()
        ml_portfolio_vol = np.sqrt(np.dot(ml_optimal_weights.T, np.dot(cov_matrix, ml_optimal_weights)))
        ml_portfolio_objective = ml_portfolio_return - self.ml_optimizer.risk_aversion * ml_portfolio_vol
        ml_portfolio_point = {
            "Return": ml_portfolio_return,
            "Volatility": ml_portfolio_vol,
            "Objective": ml_portfolio_objective
        }
        
        self.portfolio_optimizer.plot_efficient_frontier(
            portfolios_df,
            ml_portfolio_point,
            optimal_mc_weights,
            ml_optimal_weights,
            list(combined_df.columns)
        )

@log_function_call
def main():
    app = CryptoPortfolioApplication()
    app.run()

if __name__ == "__main__":
    main()
