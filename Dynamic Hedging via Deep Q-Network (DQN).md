# Dynamic Hedging via Deep Q-Network (DQN)

This repository contains a Python module demonstrating how to:
1. **Simulate asset price paths** using Geometric Brownian Motion (GBM).  
2. **Price options** using the Black-Scholes-Merton (BSM) formula.  
3. **Replicate and hedge options** via delta-hedging and a Deep Q-Network (DQN) agent.  

It includes:

- An `OptionPricer` class for Black-Scholes-Merton calculations (call price and delta).  
- A `GBMSimulator` class for simulating Geometric Brownian Motion (GBM) paths.  
- An `OptionReplicationSimulator` class to demonstrate classical delta-hedging.  
- A `HedgeSimEnv` environment to train a deep RL agent.  
- A `HedgeDQNAgent` for learning hedging strategies using Q-learning.  

## Overview

This code showcases **dynamic hedging** strategies for European call options by combining **option pricing** (Black-Scholes-Merton, Delta-Hedging) with **reinforcement learning** (a Deep Q-Network).  

### Why DQN for Hedging?
- Traditional delta-hedging rebalances based on the Black-Scholes delta at each time step.  
- A **Deep Q-Network** can learn alternative or more nuanced rebalancing strategies under various market conditions or costs, possibly improving hedging performance under certain assumptions.

**Happy Hedging!**

