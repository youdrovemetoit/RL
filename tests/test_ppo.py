import os
import sys
import unittest
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo import PPO

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
        env = PPO.make_envs("ConstantEnv", num_envs=2)
        self.addCleanup(env.close)
        obs, _ = env.reset()
        
        ep_returns_running = np.zeros(env.num_envs)
        ep_lens_running = np.zeros(env.num_envs, dtype=int)
        discounted_return = np.zeros(env.num_envs)
        
        # Get a rollout, episodes truncate after 10 (see gym.register above), which in 25 steps below is steps 9 and 19
        r, *_ = PPO.collect_rollout(env, obs, ep_returns_running, ep_lens_running, RandomActor(), ConstantCritic(10.0), steps=25, device="cpu",
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
        adv, ret = PPO.compute_gae(b["rewards"], b["values"], b["dones"], b["last_value"], gamma=0.9, lam=0.95)
        
        # Check our expected returns are always 10.0
        # Without the bootstrap, step 9 would get a return of 1, and the steps before it would fall short of 10 more and more,
        # the closer they are to the time limit
        expected_returns = torch.full_like(ret, 10.0)
        torch.testing.assert_close(ret, expected_returns)
        
        # Check the advantages
        torch.testing.assert_close(adv, torch.zeros_like(adv))
        
    def test_seeding(self):
        env_name = "CartPole-v1"
    
        class Actor(nn.Module):   
            def __init__(self, obs_space: int, action_space: int, hidden: int = 64):
                super().__init__()
                self.obs_dim = obs_space.shape[0]
                self.n_actions = action_space.n
                self.net = nn.Sequential(
                    PPO.layer_init(nn.Linear(self.obs_dim, hidden)), nn.Tanh(),
                    PPO.layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
                    # Policy head: small init so initial logits are ~uniform
                    PPO.layer_init(nn.Linear(hidden, self.n_actions), std=0.01),
                )
            def forward(self, x):
                return self.net(x)  # logits
            def dist(self, x):
                return Categorical(logits=self.forward(x))

        class Critic(nn.Module):
            def __init__(self, obs_space: int, hidden: int = 64):
                super().__init__()
                self.obs_dim = obs_space.shape[0]
                self.net = nn.Sequential(
                    PPO.layer_init(nn.Linear(self.obs_dim, hidden)), nn.Tanh(),
                    PPO.layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
                    PPO.layer_init(nn.Linear(hidden, 1), std=1.0),
                )
            def forward(self, x):
                return self.net(x).squeeze(-1)
    
        ppoTrainerConfig = PPO.PPOTrainerConfig()
        ppoTrainerConfig.num_envs = 1
        ppoTrainerConfig.seed = 0
        ppoTrainerConfig.device = "cpu"
        ppoTrainerConfig.make_actor = lambda o, a: Actor(o, a)
        ppoTrainerConfig.make_critic = lambda o: Critic(o)
    
        trainer = PPO.PPOTrainer(env_name, ppoTrainerConfig)
    
        trainConfig = PPO.PPOTrainerTrainConfig()
        trainConfig.total_steps = 4 * 512
        trainConfig.rollout_steps = 512
        trainConfig.lr = 2.5e-4
        trainConfig.adam_eps = 1e-5
        trainConfig.normalise_reward = True
        trainConfig.verbose = False
    
        ppoConfig = PPO.PPOTrainerPPOUpdateConfig()
        ppoConfig.clip_eps = 0.2
        ppoConfig.epochs = 4
        ppoConfig.minibatch_size = 64
        ppoConfig.norm_adv = True
        ppoConfig.clip_vloss = True
        ppoConfig.max_grad_norm = 0.5
        ppoConfig.ent_coef = 0.01
        ppoConfig.target_kl = 0.015
        ppoConfig.lam = 0.95    
        ppoConfig.vf_coef = 0.5
    
        # Do two runs and check they have the same output when their seeds are the same
        run = trainer.train(trainConfig, ppoConfig)['returns']
        
        trainer = PPO.PPOTrainer(env_name, ppoTrainerConfig)
        run2 = trainer.train(trainConfig, ppoConfig)['returns']
        
        self.assertNotEqual(len(run), 0)
        self.assertEqual(run, run2)

if __name__ == "__main__":
    unittest.main()
