# Market Making Game with Reinforcement Learning

## Overview
This project is a **linear market-making simulation** demonstrating optimal bid-ask spread capturing while managing inventory risk. It utilizes **Reinforcement Learning (PPO)** to optimize market-making strategies within a Gym environment.

## Key Features
- **Custom Gym Environment**: Simulates order book dynamics with bid, ask, and mid-price updates.
- **Reinforcement Learning Agent**: Trained with PPO to maximize profit while controlling inventory risk.
- **Real-Time Order Book Simulation**: Adjustable spread strategy with execution probability modeling.
- **Pygame Visualization**: Live order book rendering with bid/ask depth levels.

## Installation & Dependencies
Requires Python 3.8+ and the following dependencies:
```bash
pip install -r requirements.txt
```
Includes:
- `gym`
- `numpy`
- `pygame`
- `stable-baselines3`
  
  **Controls:**
  - **LEFT**: Widen spread
  - **DOWN**: Maintain spread
  - **RIGHT**: Narrow spread 

