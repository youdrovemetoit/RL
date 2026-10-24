import os
import sys
import unittest
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ppo.PPO

# A simple gym environment that returns 1 on every step
class ConstantEnv(gym.Env):
    
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
    
    def _get_obs(self):
        # Return value that matches the observation space
        return np.zeros(1, np.float32)
    
    def _get_info(self):
        return {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        terminated = False
        truncated = False
        reward = 1.0
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

class RandomActor(torch.nn.Module):
    def dist(self, x):
        return Categorical(logits=torch.zeros(x.shape[0], 2))

class ConstantCritic(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v
        
    def forward(self, x):
        return torch.full((x.shape[0],), self.v)

# Register the environment as PPO.make_envs creates them by id
gym.register(
    id="ConstantEnv",
    entry_point=ConstantEnv,
    max_episode_steps=10
)

class PPOTest(unittest.TestCase):
    
    def test_collect_rollout_truncation(self):
        # Create test environment
        env = ppo.PPO.make_envs("ConstantEnv", num_envs=2)
        self.addCleanup(env.close)
        obs, _ = env.reset()
        
        ep_returns_running = np.zeros(env.num_envs)
        ep_lens_running = np.zeros(env.num_envs, dtype=int)
        discounted_return = np.zeros(env.num_envs)
        
        # Get a rollout, episodes truncate after 10 (see gym.register above), which in 25 steps below is steps 9 and 19
        r, *_ = ppo.PPO.collect_rollout(env, obs, ep_returns_running, ep_lens_running, RandomActor(), ConstantCritic(10.0), steps=25, device="cpu",
            return_rms=None, discounted_return=discounted_return, gamma=0.9)
        b = r.to_tensors("cpu")
        
        # Check truncation happened at steps 9 and 19
        expected_dones = torch.zeros_like(b["dones"])
        expected_dones[[9, 19]] = 1.0
        torch.testing.assert_close(b["dones"], expected_dones)
        
        # The rewards should be 1 everywhere except the truncated steps, where we use the bootstrap-into-reward trick
        # so GAE only needs to use 'done' to cut the rollout chain, as opposed to tracking truncated and terminated separately
        expected_rewards = torch.ones_like(b["rewards"])
        expected_rewards[[9, 19]] = 1.0 + 0.9 * 10.0
        torch.testing.assert_close(b["rewards"], expected_rewards)

        # compute GAE
        adv, ret = ppo.PPO.compute_gae(b["rewards"], b["values"], b["dones"], b["last_value"], gamma=0.9, lam=0.95)
        
        # Check our expected returns are always 10.0
        # Without the bootstrap, step 9 would get a return of 1, and the steps before it would fall short of 10 more and more,
        # the closer they are to the time limit
        expected_returns = torch.full_like(ret, 10.0)
        torch.testing.assert_close(ret, expected_returns)
        
        # Check the advantages
        torch.testing.assert_close(adv, torch.zeros_like(adv))

if __name__ == "__main__":
    unittest.main()
