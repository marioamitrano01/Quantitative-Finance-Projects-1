import sys
import time
import functools
import logging
import argparse
from typing import Tuple, List

import numpy as np
import gym
from gym import spaces

import pygame


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed:.6f} seconds")
        return res
    return wrapper



class OrderBook:
    
    def __init__(self, initial_bid: float = 100.0, initial_ask: float = 100.05):
        self.bid = initial_bid
        self.ask = initial_ask
        self.mid_price = (self.bid + self.ask) / 2.0  # Always computed as (bid + ask) / 2

    @timed
    def update(self):
      
        delta = np.random.randn() * 0.1  
        self.mid_price += delta
        spread = self.ask - self.bid
        if self.mid_price < spread / 2:
            self.mid_price = spread / 2
        self.bid = self.mid_price - spread / 2
        self.ask = self.mid_price + spread / 2
        logger.debug(f"OrderBook update: bid={self.bid:.2f}, ask={self.ask:.2f}, mid={self.mid_price:.2f}")

    @timed
    def set_spread(self, new_spread: float):
        
        if self.mid_price < new_spread / 2:
            self.mid_price = new_spread / 2
        self.bid = self.mid_price - new_spread / 2
        self.ask = self.mid_price + new_spread / 2
        logger.debug(f"Set spread: {new_spread:.2f}, bid={self.bid:.2f}, ask={self.ask:.2f}")

class MarketMakingEnv(gym.Env):
    
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(MarketMakingEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.order_book = OrderBook()
        self.initial_spread = self.order_book.ask - self.order_book.bid
        self.current_spread = self.initial_spread
        self.inventory = 0.0
        self.position_value = 0.0  
        self.time_step = 0
        self.max_steps = 1000

    @timed
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.time_step += 1
        done = self.time_step >= self.max_steps

        self.order_book.update()

      
        if action == 0:
            self.current_spread *= 1.1
        elif action == 2:
            self.current_spread *= 0.9

        self.order_book.set_spread(self.current_spread)

        trade_prob = np.clip(1.0 - (self.current_spread - self.initial_spread) / self.initial_spread, 0.1, 1.0)
        trade_occurs = np.random.rand() < trade_prob
        trade_profit = 0.0

        if trade_occurs:
            trade_volume = np.random.randint(1, 10)
            profit = (self.order_book.ask - self.order_book.bid) * trade_volume
            trade_profit = profit
            if np.random.rand() < 0.5:
                self.inventory += trade_volume
            else:
                self.inventory -= trade_volume

        inventory_penalty = -0.01 * (self.inventory ** 2)
        reward = trade_profit + inventory_penalty
        self.position_value += reward

        obs = np.array([self.current_spread, self.inventory, self.order_book.mid_price], dtype=np.float32)
        info = {
            "trade_profit": trade_profit,
            "inventory_penalty": inventory_penalty,
            "position_value": self.position_value
        }
        return obs, reward, done, info

    @timed
    def reset(self) -> np.ndarray:
        self.order_book = OrderBook()
        self.initial_spread = self.order_book.ask - self.order_book.bid
        self.current_spread = self.initial_spread
        self.inventory = 0.0
        self.position_value = 0.0
        self.time_step = 0
        return np.array([self.current_spread, self.inventory, self.order_book.mid_price], dtype=np.float32)

    def render(self, mode="human"):
        print(f"Time Step: {self.time_step}")
        print(f"Mid Price: {self.order_book.mid_price:.2f}")
        print(f"Bid: {self.order_book.bid:.2f}, Ask: {self.order_book.ask:.2f}")
        print(f"Spread: {self.current_spread:.2f}, Inventory: {self.inventory}, Profit: {self.position_value:.2f}")

    def close(self):
        pass



def get_order_book_levels(env: MarketMakingEnv, num_levels: int = 10, tick: float = 0.05) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
    
    best_bid = env.order_book.bid
    best_ask = env.order_book.ask
    bid_levels = []
    ask_levels = []
    for i in range(num_levels):
        bid_price = best_bid - i * tick
        ask_price = best_ask + i * tick
        bid_volume = np.random.randint(10, 100)
        ask_volume = np.random.randint(10, 100)
        bid_levels.append((bid_price, bid_volume))
        ask_levels.append((ask_price, ask_volume))
    return bid_levels, ask_levels



class MarketMakingVisualizer:
    
    def __init__(self, mode: str = "manual", agent_model=None):
        
        pygame.init()
        self.WIDTH, self.HEIGHT = 1200, 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Mario's Market Making Simulation")
        self.font = pygame.font.SysFont("Arial", 20)
        self.clock = pygame.time.Clock()
        self.bg_color = (20, 20, 20)
        self.text_color = (240, 240, 240)
        self.bid_color = (0, 180, 0)    
        self.ask_color = (180, 0, 0)    
        self.mid_color = (200, 200, 0)  

        self.mode = mode
        self.agent_model = agent_model
        self.env = MarketMakingEnv()
        self.obs = self.env.reset()
        
        self.num_levels = 10
        self.tick = 0.05
        self.max_volume = 100
        self.orderbook_margin = 50
        self.FPS = 60 
        self.info_panel_height = 250

    def print_terminal_instructions(self):
        instructions = [
            "Mario's Trading Instructions:",
            " - RIGHT Arrow or RL Agent: Narrow the spread to increase trade frequency.",
            " - LEFT Arrow: Widen the spread to manage high inventory risk.",
            " - DOWN Arrow: Maintain the current spread."
        ]
        print("\n".join(instructions))

    def draw_interface(self):
        pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.WIDTH, self.info_panel_height))
        
        title_surf = self.font.render("Mario's Market Making Simulation", True, (255, 215, 0))
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
    
    def process_events(self) -> int:
        
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif self.mode == "manual" and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 0  # Widen spread
                elif event.key == pygame.K_DOWN:
                    action = 1  # No change
                elif event.key == pygame.K_RIGHT:
                    action = 2  # Narrow spread
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
                if done:
                    self.obs = self.env.reset()
                pygame.time.delay(100)
            
            self.draw_interface()
            self.draw_order_book()
            
            pygame.display.flip()
            self.clock.tick(self.FPS)
        
        pygame.quit()
        sys.exit()



def train_agent(total_timesteps: int = 100000):
    
    from stable_baselines3 import PPO
    env = MarketMakingEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_market_maker")
    print("Mario's training is complete. Model saved as 'ppo_market_maker.zip'")
    env.close()

def run_simulation_with_agent():
    
    from stable_baselines3 import PPO
    try:
        model = PPO.load("ppo_market_maker")
    except Exception as e:
        print("Mario's error: Could not load the model. Have you trained the agent? (Run with --train first)")
        sys.exit(1)
    visualizer = MarketMakingVisualizer(mode="agent", agent_model=model)
    visualizer.run()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train Mario's RL agent")
    parser.add_argument("--simulate", action="store_true", help="Run simulation with Mario's trained RL agent")
    parser.add_argument("--manual", action="store_true", help="Run simulation in manual mode (keyboard control)")
    args = parser.parse_args()

    if args.train:
        train_agent(total_timesteps=100000)
    elif args.simulate:
        run_simulation_with_agent()
    else:
        visualizer = MarketMakingVisualizer(mode="manual")
        visualizer.run()

if __name__ == "__main__":
    main()
