import math
import random
import re
import gc
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from functools import wraps
from abc import ABC, abstractmethod
import unittest


np.set_printoptions(suppress=True)


def log_execution(func):
    """Decorator to log function execution details."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[LOG] Starting '{func.__name__}' with args: {args[1:] if len(args)>1 else ''} kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"[LOG] Finished '{func.__name__}'")
        return result
    return wrapper

def memory_tracker(cls):
    """Class decorator to track memory cleanup when instances are deleted."""
    orig_del = getattr(cls, "__del__", None)
    def __del__(self):
        print(f"[MEMORY] Cleaning up instance of {cls.__name__}")
        if orig_del:
            orig_del(self)
    cls.__del__ = __del__
    return cls

def make_prefix_logger(prefix):
    
    def logger(message):
        print(f"{prefix}: {message}")
    return logger

module_logger = make_prefix_logger("AdvancedModule")


@memory_tracker
class OptionPricer:
   
    
    @staticmethod
    @log_execution
    def calculate_call_price(S, K, T, t, r, sigma):
        
        if not isinstance(S, (int, float)):
            raise TypeError("Asset price S must be numeric")
        S = float(S)
        dt = T - t
        if dt <= 0:
            return max(S - K, 0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * dt) / (sigma * math.sqrt(dt))
        d2 = d1 - sigma * math.sqrt(dt)
        price = (S * stats.norm.cdf(d1) -
                 K * math.exp(-r * dt) * stats.norm.cdf(d2))
        return price

    @staticmethod
    @log_execution
    def calculate_delta(S, K, T, t, r, sigma):
      
        dt = T - t
        if dt <= 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * dt) / (sigma * math.sqrt(dt))
        return stats.norm.cdf(d1)


@memory_tracker
class GBMSimulator:
    
    def __init__(self, S0, T, r, sigma, steps=100):
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.dt = T / steps

    @log_execution
    def simulate_path(self):
        path = [self.S0]
        for _ in range(1, self.steps + 1):
            noise = random.gauss(0, 1)
            drift = (self.r - 0.5 * self.sigma**2) * self.dt
            diffusion = self.sigma * math.sqrt(self.dt) * noise
            new_price = path[-1] * math.exp(drift + diffusion)
            path.append(new_price)
        return path


@memory_tracker
class Plotter:
    
    
    @staticmethod
    @log_execution
    def plot_gbm_path(path, title="Simulated GBM", x_title="Time Step", y_title="Asset Price"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(path))),
            y=path,
            mode='lines',
            line=dict(color='blue', width=2),
            name="GBM Path"
        ))
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
        fig.show()
    
    @staticmethod
    @log_execution
    def plot_delta_vs_price(price_range, delta_values, title="Delta vs. Asset Price"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_range,
            y=delta_values,
            mode='lines',
            line=dict(color='blue', width=2),
            name="Delta Curve"
        ))
        fig.update_layout(title=title, xaxis_title="Asset Price", yaxis_title="Delta")
        fig.show()
    
    @staticmethod
    @log_execution
    def plot_replication(replication_df, title="Option Replication", x_title="Time Step", y_title="Value"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=replication_df.index,
            y=replication_df['CallPrice'],
            mode='lines',
            name='Call Price',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=replication_df.index,
            y=replication_df['PortfolioValue'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
        fig.show()

    @staticmethod
    @log_execution
    def plot_histogram(data, nbins=35, title="Histogram", xaxis_title="Value", yaxis_title="Frequency"):
        fig = px.histogram(data, nbins=nbins, title=title, labels={'value': xaxis_title})
        fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)
        fig.show()

@memory_tracker
class OptionReplicationSimulator:
    
    def __init__(self, gbm_path, strike, T, r, sigma):
        self.gbm_path = gbm_path
        self.strike = strike
        self.T = T
        self.r = r
        self.sigma = sigma
        self.dt = T / (len(gbm_path) - 1)
        self.bond_values = [math.exp(self.r * i * self.dt) for i in range(len(gbm_path))]

    def replicate(self):
        replication_df = pd.DataFrame()
        def record_state(i, delta_value, bond_alloc, call_value, portfolio_value=None):
            nonlocal replication_df
            if i > 0:
                df_temp = pd.DataFrame({
                    'AssetPrice': [self.gbm_path[i]],
                    'CallPrice': [call_value],
                    'PortfolioValue': [portfolio_value],
                    'Delta': [delta_value],
                    'BondAlloc': [bond_alloc]
                })
                replication_df = pd.concat([replication_df, df_temp], ignore_index=True)

        delta_value = None
        bond_alloc = None
        for i in range(len(self.gbm_path) - 1):
            t_i = i * self.dt
            call_value = OptionPricer.calculate_call_price(self.gbm_path[i], self.strike, self.T, t_i, self.r, self.sigma)
            if i == 0:
                delta_value = OptionPricer.calculate_delta(self.gbm_path[i], self.strike, self.T, t_i, self.r, self.sigma)
                bond_alloc = (call_value - delta_value * self.gbm_path[i]) / self.bond_values[i]
                record_state(i, delta_value, bond_alloc, call_value)
            else:
                portfolio_value = delta_value * self.gbm_path[i] + bond_alloc * self.bond_values[i]
                delta_value = OptionPricer.calculate_delta(self.gbm_path[i], self.strike, self.T, t_i, self.r, self.sigma)
                bond_alloc = (call_value - delta_value * self.gbm_path[i]) / self.bond_values[i]
                record_state(i, delta_value, bond_alloc, call_value, portfolio_value)
        return replication_df


def validate_symbol(symbol):
   
    pattern = r"^[A-Z]{1,5}$"
    if not re.match(pattern, symbol):
        raise ValueError(f"Invalid symbol '{symbol}'. Must be 1 to 5 uppercase letters.")
    return symbol


@memory_tracker
class ObservationSpace:
    
    def __init__(self, n):
        self.shape = (n,)

@memory_tracker
class ActionSpace:
    
    def __init__(self, n):
        self.n = n
    
    def seed(self, seed_val):
        random.seed(seed_val)


@memory_tracker
class HedgeSimEnv:
    
    def __init__(self, S0, strike_options, T, rate_options, vol_options, steps):
        self.initial_price = S0
        self.strike_options = strike_options
        self.maturity = T
        self.rate_options = rate_options
        self.vol_options = vol_options
        self.steps = steps
        self.obs_space = ObservationSpace(8)  
        self.action_space = ActionSpace(1)
        self.portfolio_history = pd.DataFrame()
        self.episode_counter = 0
        self.asset_position = 0
        self.bond_position = 0
        self.total_reward = 0
        self.current_step = 0
        self.sim_data = None
        self.dt = None
        self.generate_simulation_data()

    @log_execution
    def generate_simulation_data(self):
        prices = [self.initial_price]
        self.strike = random.choice(list(self.strike_options))
        self.rate = random.choice(list(self.rate_options))
        self.volatility = random.choice(list(self.vol_options))
        self.dt = self.maturity / self.steps
        for _ in range(1, self.steps + 1):
            noise = random.gauss(0, 1)
            new_price = prices[-1] * math.exp((self.rate - 0.5 * self.volatility**2) * self.dt +
                                              self.volatility * math.sqrt(self.dt) * noise)
            prices.append(new_price)
        self.sim_data = pd.DataFrame(prices, columns=['Price'])
        self.sim_data['Bond'] = np.exp(self.rate * np.arange(len(self.sim_data)) * self.dt)

    @log_execution
    def get_state(self):
        current_price = self.sim_data['Price'].iloc[self.current_step]
        current_bond = self.sim_data['Bond'].iloc[self.current_step]
        time_remaining = self.maturity - self.current_step * self.dt
        call_price = (OptionPricer.calculate_call_price(current_price, self.strike, self.maturity,
                                                          self.current_step * self.dt, self.rate, self.volatility)
                      if time_remaining > 0 else max(current_price - self.strike, 0))
        state = np.array([current_price, current_bond, time_remaining, call_price,
                          self.strike, self.rate, self.asset_position, self.bond_position])
        return state, {}

    def seed(self, seed_val=None):
        if seed_val is not None:
            random.seed(seed_val)

    @log_execution
    def reset(self):
        self.current_step = 0
        self.asset_position = 0
        self.bond_position = 0
        self.total_reward = 0
        self.episode_counter += 1
        self.generate_simulation_data()
        state, info = self.get_state()
        return state, info

    @log_execution
    def step(self, action):
        if self.current_step == 0:
            reward = 0.0
            self.current_step += 1
            self.asset_position = float(action)
            state, _ = self.get_state()
            self.bond_position = (state[3] - self.asset_position * state[0]) / state[1]
            new_state, _ = self.get_state()
        else:
            self.current_step += 1
            new_state, _ = self.get_state()
            portfolio_value = self.asset_position * new_state[0] + self.bond_position * new_state[1]
            pnl = portfolio_value - new_state[3]
            record = pd.DataFrame({
                'Episode': [self.episode_counter],
                'AssetPos': [self.asset_position],
                'BondPos': [self.bond_position],
                'PortfolioVal': [portfolio_value],
                'CallPrice': [new_state[3]],
                'P&L[$]': [pnl],
                'P&L[%]': [pnl / (new_state[3] if new_state[3] > 1e-4 else 1e-4) * 100],
                'Price': [new_state[0]],
                'Bond': [new_state[1]],
                'Strike': [self.strike],
                'Rate': [self.rate],
                'Volatility': [self.volatility]
            })
            self.portfolio_history = pd.concat([self.portfolio_history, record], ignore_index=True)
            reward = - (pnl) ** 2
            self.asset_position = float(action)
            self.bond_position = (new_state[3] - self.asset_position * new_state[0]) / new_state[1]
        done = (self.current_step == len(self.sim_data) - 1)
        self.state = new_state
        return new_state, float(reward), done, False, {}


@memory_tracker
class BaseDQNAgent(ABC):
    """
    class for Deep Q-Network agents.
    """
    def __init__(self, symbol, features, n_features, environment, hidden_units, learning_rate):
        self.symbol = validate_symbol(symbol)
        self.features = features
        self.n_features = n_features
        self.env = environment
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.memory = []
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32
        self.build_model(hidden_units, learning_rate)

    def reshape_state(self, state):
        return np.reshape(state, (1, -1))

    def store_experience(self, state, action, next_state, reward, done):
        self.memory.append((self.reshape_state(state), action, self.reshape_state(next_state), reward, done))

    def learn(self, episodes):
        for e in range(episodes):
            state, _ = self.env.reset()
            state = self.reshape_state(state)
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.reshape_state(next_state)
                self.store_experience(state, action, next_state, reward, done)
                state = next_state
                self.replay()

    @abstractmethod
    def act(self, state):
        raise NotImplementedError("Method 'act' must be implemented by subclasses.")

    @abstractmethod
    def replay(self):
        raise NotImplementedError("Method 'replay' must be implemented by subclasses.")


@memory_tracker
class HedgeDQNAgent(BaseDQNAgent):
    """
    Deep Q-Learning agent for option hedging.
    """
    @log_execution
    def build_model(self, hidden_units, learning_rate):
        self.model = Sequential()
        self.model.add(Input(shape=(self.n_features,)))
        self.model.add(Dense(hidden_units, activation='relu'))
        self.model.add(Dense(hidden_units, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    @log_execution
    def optimize_action(self, state):
        bounds = [(0, 1)]
        def objective(x):
            temp_state = state.copy()
            temp_state[0, 6] = x  
            temp_state[0, 7] = (temp_state[0, 3] - x * temp_state[0, 0]) / temp_state[0, 1]  # Bond position
            return self.model.predict(temp_state)[0, 0]
        try:
            res = minimize(lambda x: -objective(x), 0.5, bounds=bounds, method='Powell')
            action_val = res.x[0]
        except Exception as e:
            module_logger(f"Optimization exception: {e}. Defaulting to current asset position.")
            action_val = self.env.asset_position
        return action_val

    def act(self, state):
        return self.env.action_space.sample() if random.random() <= self.epsilon else self.optimize_action(state)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, done in batch:
            target = reward
            if not done:
                ns = next_state.copy()
                next_action = self.optimize_action(ns)
                ns[0, 6] = next_action
                ns[0, 7] = (ns[0, 3] - next_action * ns[0, 0]) / ns[0, 1]
                target += self.gamma * self.model.predict(ns)[0, 0]
            self.model.fit(state, np.array([target]), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def test(self, episodes, verbose=True):
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = self.reshape_state(state)
            total_reward = 0
            while True:
                action = self.optimize_action(state)
                state, reward, done, _, _ = self.env.step(action)
                state = self.reshape_state(state)
                total_reward += reward
                if done:
                    if verbose:
                        print(f"Episode {e}: Total Penalty = {total_reward:4.2f}")
                    break

class TestOptionPricer(unittest.TestCase):
    def test_call_price(self):
        price = OptionPricer.calculate_call_price(100, 100, 1.0, 0.0, 0.04, 0.2)
        self.assertAlmostEqual(price, 10.4506, places=2)
    
    def test_delta(self):
        delta = OptionPricer.calculate_delta(100, 100, 1.0, 0.0, 0.04, 0.2)
        self.assertTrue(0.5 < delta < 0.7)


def main():
    module_logger("Starting option hedging simulation.")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOptionPricer)
    unittest.TextTestRunner(verbosity=2).run(suite)

    
    initial_price = 100
    strike_price = 100
    maturity = 1.0
    current_time = 0.0
    risk_free_rate = 0.04
    volatility = 0.2
    call_price = OptionPricer.calculate_call_price(initial_price, strike_price, maturity, current_time, risk_free_rate, volatility)
    print("Call Option Price:", call_price)

    gbm_sim = GBMSimulator(initial_price, maturity, risk_free_rate, volatility, steps=100)
    gbm_path = gbm_sim.simulate_path()
    Plotter.plot_gbm_path(gbm_path)

    price_range = list(range(40, 181, 4))
    delta_list = [OptionPricer.calculate_delta(price, strike_price, maturity, 0, risk_free_rate, volatility) for price in price_range]
    Plotter.plot_delta_vs_price(price_range, delta_list)

    replication_sim = OptionReplicationSimulator(gbm_path, strike_price, maturity, risk_free_rate, volatility)
    replication_results = replication_sim.replicate()
    Plotter.plot_replication(replication_results)
    mean_pl = (replication_results['PortfolioValue'] - replication_results['CallPrice']).mean()
    mse_pl = ((replication_results['PortfolioValue'] - replication_results['CallPrice']) ** 2).mean()
    print("Mean P&L:", mean_pl)
    print("Mean Squared Error:", mse_pl)
    Plotter.plot_histogram(replication_results['PortfolioValue'] - replication_results['CallPrice'], title="P&L Histogram")

    env = HedgeSimEnv(
        S0=100.0,
        strike_options=np.array([0.9, 0.95, 1.0, 1.05, 1.10]) * initial_price,
        T=1.0,
        rate_options=[0, 0.01, 0.05],
        vol_options=[0.1, 0.15, 0.2],
        steps=100 * 252
    )
    env.seed(750)
    env.generate_simulation_data()

    norm_asset = env.sim_data['Price'] / env.sim_data['Price'].iloc[0]
    norm_bond = env.sim_data['Bond'] / env.sim_data['Bond'].iloc[0]
    fig_env = go.Figure()
    fig_env.add_trace(go.Scatter(
        x=env.sim_data.index, y=norm_asset,
        mode='lines', name='Asset',
        line=dict(color='red', dash='dot')
    ))
    fig_env.add_trace(go.Scatter(
        x=env.sim_data.index, y=norm_bond,
        mode='lines', name='Bond',
        line=dict(color='blue', dash='dash')
    ))
    fig_env.update_layout(title="Hedging Simulation Data", xaxis_title="Time Step", yaxis_title="Normalized Price")
    fig_env.show()

    env.reset()
    for _ in range(env.steps - 1):
        env.step(env.action_space.sample())
    print("First few portfolio entries:")
    print(env.portfolio_history.head().round(4))

    fig_portfolio = go.Figure()
    fig_portfolio.add_trace(go.Scatter(
        x=env.portfolio_history.index, y=env.portfolio_history['CallPrice'],
        mode='lines', name='Call Price', line=dict(color='red', dash='dot')
    ))
    fig_portfolio.add_trace(go.Scatter(
        x=env.portfolio_history.index, y=env.portfolio_history['PortfolioVal'],
        mode='lines', name='Portfolio Value', line=dict(color='blue')
    ))
    fig_portfolio.update_layout(title="Option Replication vs. Hedging", xaxis_title="Time Step", yaxis_title="Value")
    fig_portfolio.show()

    total_abs_pnl = env.portfolio_history['P&L[$]'].abs().sum()
    print("Total Absolute P&L:", total_abs_pnl)
    Plotter.plot_histogram(env.portfolio_history['P&L[$]'], title="Hedging P&L Histogram")

    
    random.seed(100)
    np.random.seed(100)
    tf.random.set_seed(100)

    agent = HedgeDQNAgent(
        symbol='SYM', features=None, n_features=8,
        environment=env, hidden_units=128, learning_rate=0.0001
    )
    training_episodes = 250
    print("Training HedgeDQNAgent...")
    agent.learn(training_episodes)
    print("Epsilon after training:", agent.epsilon)
    print("Testing HedgeDQNAgent...")
    agent.test(10)

    
    final_episode = int(agent.env.portfolio_history['Episode'].max()) - 1
    print("Episode", final_episode, "P&L Statistics:")
    print(agent.env.portfolio_history[agent.env.portfolio_history['Episode'] == final_episode]['P&L[$]'].describe())
    p = agent.env.portfolio_history[agent.env.portfolio_history['Episode'] == final_episode].iloc[0][['Strike', 'Rate', 'Volatility']]
    plot_title = f"CALL | Strike={p['Strike']:.1f} | Rate={p['Rate']} | Volatility={p['Volatility']}"

    final_df = agent.env.portfolio_history[agent.env.portfolio_history['Episode'] == final_episode].iloc[:100]
    fig_final = make_subplots(specs=[[{"secondary_y": True}]])
    fig_final.add_trace(go.Scatter(
        x=final_df.index, y=final_df['PortfolioVal'],
        mode='lines', name='Portfolio Value', line=dict(color='red')
    ), secondary_y=False)
    fig_final.add_trace(go.Scatter(
        x=final_df.index, y=final_df['CallPrice'],
        mode='lines', name='Call Price', line=dict(color='blue', dash='dot')
    ), secondary_y=False)
    fig_final.add_trace(go.Scatter(
        x=final_df.index, y=final_df['Price'],
        mode='lines', name='Asset Price', line=dict(color='green', dash='dot')
    ), secondary_y=True)
    fig_final.update_layout(title=plot_title, xaxis_title="Time Step", legend=dict(x=0.01, y=0.99))
    fig_final.update_yaxes(title_text="Value", secondary_y=False)
    fig_final.update_yaxes(title_text="Asset Price", secondary_y=True)
    fig_final.show()

   
    final_pl = agent.env.portfolio_history[agent.env.portfolio_history['Episode'] == final_episode]['P&L[$]']
    fig_final_hist = px.histogram(final_pl, nbins=35, title=plot_title, labels={'value': 'P&L'})
    fig_final_hist.update_layout(xaxis_title="P&L", yaxis_title="Frequency")
    fig_final_hist.show()

    module_logger("Simulation complete. Initiating memory cleanup.")
    gc.collect()

if __name__ == "__main__":
    main()
