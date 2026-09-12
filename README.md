# Reinforcement Learning

This repository contains implementations of foundation RL algorithms, developed upwards from base principals. All are developed from scratch to develop an understanding of the processes involved, and should provide a path to understanding modern RL techiques.

## Overview

### Monte Carlo

The very basics, a notebook that samples episodes from the MDP and uses it to determine the value function for the state space. Demonstrated on the FrozenLake gym environment.

### SARSA $\lambda$

An implementation of SARSA

### Policy Gradient (REINFORCE)



### Q-Learning

### TD-Learning



### DQN

### Actor Critic (A2C)

### Proximal Policy Optimisation

Implementation of [PPO](https://arxiv.org/abs/1707.06347), with a number of the optimisations that are typically required to make it function well.

I've shifted most of the code to a Python file that is called by the example notebook. As the implementation becomes more useful than the previous learning notebooks, this will make it easier to experiment and extend to practical use cases.
