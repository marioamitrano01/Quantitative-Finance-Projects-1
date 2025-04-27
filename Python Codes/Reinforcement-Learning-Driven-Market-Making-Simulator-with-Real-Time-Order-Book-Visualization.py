import sys
import time
import functools
import logging
import argparse
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import gym
from gym import spaces

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable_baselines3 not available. Training features will be disabled.")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("market_maker")


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {elapsed:.6f} seconds")
        return res
    return wrapper


@dataclass
class OrderBookConfig:
    initial_bid: float = 100.0
    initial_ask: float = 100.05
    volatility: float = 0.03  
    min_price: float = 1.0


class OrderBook:
    
    def __init__(self, config: Optional[OrderBookConfig] = None):
        
        self.config = config or OrderBookConfig()
        self.bid = self.config.initial_bid
        self.ask = self.config.initial_ask
        self.mid_price = (self.bid + self.ask) / 2.0
        self.volatility = self.config.volatility
        self.min_price = self.config.min_price

    @timed
    def update(self) -> None:
        delta = np.random.randn() * self.volatility
        
        if abs(delta) > self.volatility * 2:
            delta = np.sign(delta) * self.volatility * 2
            
        self.mid_price += delta
        
        if self.mid_price < self.min_price:
            self.mid_price = self.min_price
            
        spread = self.ask - self.bid
        self.bid = self.mid_price - spread / 2
        self.ask = self.mid_price + spread / 2
        logger.debug(f"OrderBook update: bid={self.bid:.2f}, ask={self.ask:.2f}, mid={self.mid_price:.2f}")

    @timed
    def set_spread(self, new_spread: float) -> None:
       
        if new_spread <= 0:
            logger.warning(f"Invalid spread value: {new_spread}. Using minimum of 0.01")
            new_spread = 0.01
            
        if self.mid_price < new_spread / 2:
            self.mid_price = new_spread / 2
            
        self.bid = self.mid_price - new_spread / 2
        self.ask = self.mid_price + new_spread / 2
        logger.debug(f"Set spread: {new_spread:.2f}, bid={self.bid:.2f}, ask={self.ask:.2f}")


@dataclass
class EnvConfig:
    max_steps: int = 1000
    inventory_penalty_factor: float = 0.01
    min_trade_probability: float = 0.1
    max_trade_volume: int = 5  
    price_update_frequency: int = 3  
    order_book_config: OrderBookConfig = OrderBookConfig()


class MarketMakingEnv(gym.Env):
    
    metadata = {"render.modes": ["human", "terminal"]}

    def __init__(self, config: Optional[EnvConfig] = None):
        
        super(MarketMakingEnv, self).__init__()
        
        self.config = config or EnvConfig()
        
        self.action_space = spaces.Discrete(3)  
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        
        self.order_book = OrderBook(self.config.order_book_config)
        self.reset()

    @timed
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        
        self.time_step += 1
        done = self.time_step >= self.config.max_steps

        if self.time_step % self.config.price_update_frequency == 0:
            self.order_book.update()

        if action == 0:  
            self.current_spread *= 1.1
        elif action == 2:  
            self.current_spread *= 0.9

        self.order_book.set_spread(self.current_spread)

        trade_prob = np.clip(
            1.0 - (self.current_spread - self.initial_spread) / self.initial_spread,
            self.config.min_trade_probability, 
            1.0
        )
        
        trade_occurs = np.random.rand() < trade_prob
        trade_profit = 0.0
        trade_volume = 0

        if trade_occurs:
            trade_volume = np.random.randint(1, self.config.max_trade_volume)
            profit = (self.order_book.ask - self.order_book.bid) * trade_volume
            trade_profit = profit
            
            if np.random.rand() < 0.5:
                self.inventory += trade_volume
            else:
                self.inventory -= trade_volume

        inventory_penalty = -self.config.inventory_penalty_factor * (self.inventory ** 2)
        reward = trade_profit + inventory_penalty
        self.position_value += reward

        obs = np.array([
            self.current_spread, 
            self.inventory, 
            self.order_book.mid_price
        ], dtype=np.float32)
        
        info = {
            "trade_profit": trade_profit,
            "trade_volume": trade_volume,
            "trade_occurred": trade_occurs,
            "inventory_penalty": inventory_penalty,
            "position_value": self.position_value,
            "mid_price": self.order_book.mid_price
        }
        
        return obs, reward, done, info

    @timed
    def reset(self) -> np.ndarray:
       
        self.order_book = OrderBook(self.config.order_book_config)
        self.initial_spread = self.order_book.ask - self.order_book.bid
        self.current_spread = self.initial_spread
        self.inventory = 0.0
        self.position_value = 0.0
        self.time_step = 0
        
        return np.array([
            self.current_spread, 
            self.inventory, 
            self.order_book.mid_price
        ], dtype=np.float32)

    def render(self, mode="human"):
        
        if mode == "terminal":
            print(f"Time Step: {self.time_step}")
            print(f"Mid Price: {self.order_book.mid_price:.2f}")
            print(f"Bid: {self.order_book.bid:.2f}, Ask: {self.order_book.ask:.2f}")
            print(f"Spread: {self.current_spread:.2f}, Inventory: {self.inventory}, Profit: {self.position_value:.2f}")
        elif mode == "human" and not PYGAME_AVAILABLE:
            print("Pygame not available. Use mode='terminal' instead.")
            self.render(mode="terminal")

    def close(self):
        pass


def get_order_book_levels(
    env: MarketMakingEnv, 
    num_levels: int = 10, 
    tick: float = 0.05
) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
    
    best_bid = env.order_book.bid
    best_ask = env.order_book.ask
    bid_levels = []
    ask_levels = []
    
    for i in range(num_levels):
        bid_price = best_bid - i * tick
        ask_price = best_ask + i * tick
        
        volume_factor = max(0.3, 1.0 - i * 0.07)
        bid_volume = int(np.random.randint(10, 100) * volume_factor)
        ask_volume = int(np.random.randint(10, 100) * volume_factor)
        
        bid_levels.append((bid_price, bid_volume))
        ask_levels.append((ask_price, ask_volume))
        
    return bid_levels, ask_levels


class MarketMakingVisualizer:
    
    
    def __init__(self, mode: str = "manual", agent_model=None, config: Optional[EnvConfig] = None):
        
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame is required for visualization. Please install it.")
            
        pygame.init()
        self.WIDTH, self.HEIGHT = 1200, 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Mario's Market Making Simulation")
        self.font = pygame.font.SysFont("Arial", 20)
        self.title_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.clock = pygame.time.Clock()
        
        self.bg_color = (20, 20, 20)
        self.text_color = (240, 240, 240)
        self.bid_color = (0, 180, 0)    
        self.ask_color = (180, 0, 0)    
        self.mid_color = (200, 200, 0)
        self.panel_color = (30, 30, 30)
        self.title_color = (255, 215, 0)
        
        self.mode = mode
        self.agent_model = agent_model
        self.env = MarketMakingEnv(config)
        self.obs = self.env.reset()
        
        self.num_levels = 10
        self.tick = 0.05
        self.max_volume = 100
        self.orderbook_margin = 50
        self.FPS = 30  
        self.info_panel_height = 250
        
        self.trade_history = []  
        self.price_history = []  
        self.action_history = []  
        
        self.icons = {}
        self.setup_icons()

    def setup_icons(self):
        """Create simple icons using pygame shapes."""
        widen = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.polygon(widen, (200, 100, 100), [(15, 5), (5, 15), (25, 15)])
        pygame.draw.polygon(widen, (200, 100, 100), [(15, 25), (5, 15), (25, 15)])
        
        narrow = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.polygon(narrow, (100, 200, 100), [(5, 5), (25, 5), (15, 15)])
        pygame.draw.polygon(narrow, (100, 200, 100), [(5, 25), (25, 25), (15, 15)])
        
        maintain = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.rect(maintain, (150, 150, 200), (5, 10, 20, 10))
        
        self.icons = {
            "widen": widen,
            "narrow": narrow,
            "maintain": maintain
        }

    def print_terminal_instructions(self):
        instructions = [
            "Mario's Trading Instructions:",
            " - RIGHT Arrow or RL Agent: Narrow the spread to increase trade frequency.",
            " - LEFT Arrow: Widen the spread to manage high inventory risk.",
            " - DOWN Arrow: Maintain the current spread.",
            " - ESC: Exit simulation",
            "",
            "Note: Order book updates have been slowed down for better visibility."
        ]
        print("\n".join(instructions))

    def draw_interface(self):
        self.screen.fill(self.bg_color)
        
        pygame.draw.rect(self.screen, self.panel_color, (0, 0, self.WIDTH, self.info_panel_height))
        
        title_surf = self.title_font.render("Mario's Market Making Simulation", True, self.title_color)
        self.screen.blit(title_surf, (self.WIDTH // 2 - title_surf.get_width() // 2, 10))
        
        info_text = [
            f"Time Step: {self.env.time_step}",
            f"Bid: {self.env.order_book.bid:.2f}",
            f"Ask: {self.env.order_book.ask:.2f}",
            f"Mid Price: {self.env.order_book.mid_price:.2f}",
            f"Spread: {self.env.current_spread:.2f}",
            f"Inventory: {self.env.inventory}",
            f"Profit: {self.env.position_value:.2f}",
            f"Mode: " + ("Manual" if self.mode == "manual" else "RL Agent")
        ]
        
        for idx, line in enumerate(info_text):
            text_surf = self.font.render(line, True, self.text_color)
            self.screen.blit(text_surf, (self.orderbook_margin, 50 + idx * 25))
        
        if self.action_history:
            action_text = "Last Action: "
            if self.action_history[-1] == 0:
                action_text += "Widen Spread"
                icon = self.icons["widen"]
            elif self.action_history[-1] == 1:
                action_text += "Maintain Spread"
                icon = self.icons["maintain"]
            else:
                action_text += "Narrow Spread"
                icon = self.icons["narrow"]
                
            action_surf = self.font.render(action_text, True, self.text_color)
            self.screen.blit(action_surf, (self.WIDTH - 300, 50))
            self.screen.blit(icon, (self.WIDTH - 300 + action_surf.get_width() + 10, 50))
    
    def draw_order_book(self):
        bid_levels, ask_levels = get_order_book_levels(self.env, self.num_levels, self.tick)
        
        level_height = 30
        max_bar_width = 400
        bid_x = self.orderbook_margin
        bid_y_start = self.info_panel_height + 50
        
        for i, (price, volume) in enumerate(bid_levels):
            bar_width = int((volume / self.max_volume) * max_bar_width)
            y_pos = bid_y_start + i * (level_height + 5)
            bid_rect = pygame.Rect(bid_x, y_pos, bar_width, level_height)
            pygame.draw.rect(self.screen, self.bid_color, bid_rect)
            pygame.draw.rect(self.screen, self.text_color, bid_rect, 2)
            text_line = self.font.render(f"{price:.2f} | Vol: {volume}", True, self.text_color)
            self.screen.blit(text_line, (bid_x + bar_width + 10, y_pos + 5))
        
        ask_x_right = self.WIDTH - self.orderbook_margin
        ask_y_start = self.info_panel_height + 50
        for i, (price, volume) in enumerate(ask_levels):
            bar_width = int((volume / self.max_volume) * max_bar_width)
            y_pos = ask_y_start + i * (level_height + 5)
            ask_rect = pygame.Rect(ask_x_right - bar_width, y_pos, bar_width, level_height)
            pygame.draw.rect(self.screen, self.ask_color, ask_rect)
            pygame.draw.rect(self.screen, self.text_color, ask_rect, 2)
            text_line = self.font.render(f"Vol: {volume} | {price:.2f}", True, self.text_color)
            self.screen.blit(text_line, (ask_x_right - bar_width - 150, y_pos + 5))
        
        mid_line_y = self.HEIGHT // 2
        pygame.draw.line(self.screen, self.mid_color, (0, mid_line_y), (self.WIDTH, mid_line_y), 2)
        mid_text = self.font.render(f"Mid Price: {self.env.order_book.mid_price:.2f}", True, self.mid_color)
        self.screen.blit(mid_text, (self.WIDTH // 2 - mid_text.get_width() // 2, mid_line_y - 30))
    
    def draw_price_chart(self):
        self.price_history.append(self.env.order_book.mid_price)
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
            
        chart_width = 300
        chart_height = 150
        chart_x = self.WIDTH - chart_width - 50
        chart_y = 80
        
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (chart_x, chart_y, chart_width, chart_height))
        pygame.draw.rect(self.screen, self.text_color, 
                        (chart_x, chart_y, chart_width, chart_height), 1)
        
        if len(self.price_history) > 1:
            min_price = min(self.price_history)
            max_price = max(self.price_history)
            price_range = max(max_price - min_price, 0.01)  
            
            points = []
            for i, price in enumerate(self.price_history):
                x = chart_x + (i / (len(self.price_history) - 1)) * chart_width
                y = chart_y + chart_height - ((price - min_price) / price_range) * chart_height
                points.append((x, y))
                
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.mid_color, False, points, 2)
        
        chart_title = self.font.render("Price History", True, self.text_color)
        self.screen.blit(chart_title, (chart_x, chart_y - 25))

    def process_events(self) -> Optional[int]:
        
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif self.mode == "manual":
                    if event.key == pygame.K_LEFT:
                        action = 0 
                    elif event.key == pygame.K_DOWN:
                        action = 1  
                    elif event.key == pygame.K_RIGHT:
                        action = 2  
        return action
    
    def run(self):
        self.print_terminal_instructions()
        
        running = True
        while running:
            action = self.process_events()
            
            if self.mode == "agent" and action is None and self.agent_model is not None:
                action, _ = self.agent_model.predict(self.obs, deterministic=True)
            
            if action is not None:
                self.obs, reward, done, info = self.env.step(action)
                self.action_history.append(action)
                
                if done:
                    self.obs = self.env.reset()
                    self.price_history = []
                    self.action_history = []
                    
                pygame.time.delay(300)
            
            self.draw_interface()
            self.draw_price_chart()
            self.draw_order_book()
            
            pygame.display.flip()
            self.clock.tick(self.FPS)
        
        pygame.quit()
        sys.exit()


def train_agent(total_timesteps: int = 100000, save_path: str = "models/ppo_market_maker"):
    
    if not SB3_AVAILABLE:
        print("Error: stable_baselines3 is required for training. Install with pip install stable-baselines3")
        sys.exit(1)
        
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    env_config = EnvConfig(
        max_steps=1000,
        inventory_penalty_factor=0.01,
        min_trade_probability=0.1
    )
    
    env = MarketMakingEnv(config=env_config)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./tensorboard_logs/"
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    model.save(save_path)
    print(f"Mario's training is complete. Model saved as '{save_path}'")
    env.close()


def run_simulation_with_agent(model_path: str = "models/ppo_market_maker"):
    
    if not SB3_AVAILABLE:
        print("Error: stable_baselines3 is required for agent simulation.")
        sys.exit(1)
        
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Mario's error: Could not load the model: {e}")
        print("Have you trained the agent? (Run with --train first)")
        sys.exit(1)
        
    env_config = EnvConfig(max_steps=2000) 
    
    visualizer = MarketMakingVisualizer(mode="agent", agent_model=model, config=env_config)
    visualizer.run()


def main():
    parser = argparse.ArgumentParser(description="Mario's Market Making Simulation")
    
    parser.add_argument("train", action="store_true", help="Train Mario's RL agent")
    parser.add_argument("simulate", action="store_true", help="Run simulation with Mario's trained RL agent")
    parser.add_argument("manual", action="store_true", help="Run simulation in manual mode (keyboard control)")
    parser.add_argument("timesteps", type=int, default=100000, help="Number of timesteps for training")
    parser.add_argument("model-path", type=str, default="models/ppo_market_maker", 
                        help="Path to save/load model")
    parser.add_argument("log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    logger.setLevel(getattr(logging, args.log_level))
    
    if args.train:
        train_agent(total_timesteps=args.timesteps, save_path=args.model_path)
    elif args.simulate:
        run_simulation_with_agent(model_path=args.model_path)
    elif args.manual:
        env_config = EnvConfig(max_steps=2000)  
        visualizer = MarketMakingVisualizer(mode="manual", config=env_config)
        visualizer.run()
    else:
        env_config = EnvConfig(max_steps=2000)
        visualizer = MarketMakingVisualizer(mode="manual", config=env_config)
        visualizer.run()


if __name__ == "__main__":
    main()
