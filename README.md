# Reinforcement Learning

This repository contains implementations of foundation RL algorithms, developed upwards from base principals. All are developed from scratch to develop an understanding of the processes involved, and should provide a path to understanding modern RL techiques.

## Overview

### Monte Carlo

The very basics, a notebook that samples episodes from the MDP and uses it to determine the value function for the state space. Demonstrated on the FrozenLake gym environment.

### TD-Learning

Implementation of Temporal Difference Learning as opposed to the MC technique above.

### SARSA $\lambda$

An implementation of SARSA-$\lambda$ using eligibility traces. The notebook demonstrates the algorithm on grid world environments with no deep learning.

### Q-Learning

An implementation of Q-Learning, essentially using the same notebook as with SARSA-$\lambda$, updated to use the new algorithm.

### DQN

Introducing some deep learning into the mix now, with this Deep Q-Learning notebook showing how the Q-Learning algorithm can be adapted to use neural networks to increase its flexibility.

### Policy Gradient (REINFORCE)

Implementation of the most basic policy optimisation algorithm.

### Actor Critic (A2C)

Implementation of the Actor-Critic algorithm, leading us on to PPO.

### Proximal Policy Optimisation

Implementation of [PPO](https://arxiv.org/abs/1707.06347), with a number of the optimisations that are typically required to make it function well.

I've shifted most of the code to a Python file that is called by the example notebook. As the implementation becomes more useful than the previous learning notebooks, this will make it easier to experiment and extend to practical use cases.
