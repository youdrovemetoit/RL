import os
import sys
import tempfile
import unittest
from typing import Optional
import warnings

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo import ppo

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

# Register the environment as ppo.make_envs creates them by id
gym.register(
    id="ConstantEnv",
    entry_point=ConstantEnv,
    max_episode_steps=10
)

class Actor(nn.Module):   
    def __init__(self, obs_space: int, action_space: int, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_space.shape[0]
        self.n_actions = action_space.n
        self.net = nn.Sequential(
            ppo.layer_init(nn.Linear(self.obs_dim, hidden)), nn.Tanh(),
            ppo.layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            # Policy head: small init so initial logits are ~uniform
            ppo.layer_init(nn.Linear(hidden, self.n_actions), std=0.01),
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
            ppo.layer_init(nn.Linear(self.obs_dim, hidden)), nn.Tanh(),
            ppo.layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            ppo.layer_init(nn.Linear(hidden, 1), std=1.0),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class PPOTest(unittest.TestCase):
    
    def test_collect_rollout_truncation(self):
        # Create test environment
        env = ppo.make_envs("ConstantEnv", num_envs=2)
        self.addCleanup(env.close)
        obs, _ = env.reset()
        
        ep_returns_running = np.zeros(env.num_envs)
        ep_lens_running = np.zeros(env.num_envs, dtype=int)
        discounted_return = np.zeros(env.num_envs)
        
        # Get a rollout, episodes truncate after 10 (see gym.register above), which in 25 steps below is steps 9 and 19
        r, *_ = ppo.collect_rollout(env, obs, ep_returns_running, ep_lens_running, RandomActor(), ConstantCritic(10.0), steps=25, device="cpu",
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
        adv, ret = ppo.compute_gae(b["rewards"], b["values"], b["dones"], b["last_value"], gamma=0.9, lam=0.95)
        
        # Check our expected returns are always 10.0
        # Without the bootstrap, step 9 would get a return of 1, and the steps before it would fall short of 10 more and more,
        # the closer they are to the time limit
        expected_returns = torch.full_like(ret, 10.0)
        torch.testing.assert_close(ret, expected_returns)
        
        # Check the advantages
        torch.testing.assert_close(adv, torch.zeros_like(adv))
        
    def test_seeding(self):
        env_name = "CartPole-v1"
    
        ppoTrainerConfig = ppo.PPOTrainerConfig()
        ppoTrainerConfig.num_envs = 1
        ppoTrainerConfig.seed = 0
        ppoTrainerConfig.device = "cpu"
        ppoTrainerConfig.make_actor = lambda o, a: Actor(o, a)
        ppoTrainerConfig.make_critic = lambda o: Critic(o)
    
        trainer = ppo.PPOTrainer(env_name, ppoTrainerConfig)
    
        trainConfig = ppo.PPOTrainerTrainConfig()
        trainConfig.total_steps = 4 * 512
        trainConfig.rollout_steps = 512
        trainConfig.lr = 2.5e-4
        trainConfig.adam_eps = 1e-5
        trainConfig.normalise_reward = True
        trainConfig.verbose = False
    
        ppoConfig = ppo.PPOTrainerPPOUpdateConfig()
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
        
        trainer = ppo.PPOTrainer(env_name, ppoTrainerConfig)
        run2 = trainer.train(trainConfig, ppoConfig)['returns']
        
        self.assertNotEqual(len(run), 0)
        self.assertEqual(run, run2)

    def make_trainer(self, run_dir: str, seed: int=0) -> ppo.PPOTrainer:
        config = ppo.PPOTrainerConfig()
        config.num_envs = 2
        config.seed = seed
        config.run_dir = run_dir
        config.make_actor = lambda o, a: Actor(o, a)
        config.make_critic = lambda o: Critic(o)
        return ppo.PPOTrainer("CartPole-v1", config)
    
    def make_train_config(self, total_steps: int, norm_obs: bool, norm_reward: bool, checkpoint_every: int):
        train_config = ppo.PPOTrainerTrainConfig()
        train_config.total_steps = total_steps
        train_config.rollout_steps = 64
        train_config.verbose = False
        train_config.normalise_observations = norm_obs
        train_config.normalise_reward = norm_reward
        train_config.checkpoint_every = checkpoint_every
        return train_config
    
    def assertStateDictEqual(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)
    
    def test_resume(self):
        STEP = 64 * 2  # steps per update
        
        ppo_config = ppo.PPOTrainerPPOUpdateConfig()
        for norm_obs, norm_reward in [(False, False), (True, True)]:
            with self.subTest(norm_obs=norm_obs, norm_reward=norm_reward), tempfile.TemporaryDirectory() as tmp:
                ckpt_dir = lambda run: os.path.join(tmp, run, "checkpoints", "CartPole-v1", "seed_0")
                
                # Base run: 4 updates, with snapshots at updates 2 and 4
                base = self.make_trainer(os.path.join(tmp, "base"))
                base_info = base.train(self.make_train_config(4*STEP, norm_obs, norm_reward, 2), ppo_config)
                self.assertFalse(base_info["resumed"])
                mid_path = os.path.join(ckpt_dir("base"), f"checkpoint_{2*STEP:09d}.pt")
                # Load data stored at midpoint checkpoint
                mid = torch.load(mid_path, weights_only=False)
                
                # 1. Check loading restores state: resuming with the same total results in 0 updates run
                final_path = os.path.join(ckpt_dir("base"), f"checkpoint_{4*STEP:09d}.pt")
                final = torch.load(final_path, weights_only=False)

                loaded = self.make_trainer(os.path.join(tmp, "loaded"))
                loaded_info = loaded.train(self.make_train_config(4*STEP, norm_obs, norm_reward, None), ppo_config, resume_from=final_path)
                self.assertTrue(loaded_info["resumed"])
                loaded.checkpoint_manager._load_checkpoint(final_path) # re-load so nothing from train()'s env.reset is mixed in
                self.assertStateDictEqual(loaded.actor.state_dict(), final["actor"])
                self.assertStateDictEqual(loaded.critic.state_dict(), final["critic"])
                if norm_obs:
                    np.testing.assert_array_equal(loaded.obs_rms.mean, final["obs_rms"]["mean"])
                    
                if norm_obs:
                    np.testing.assert_array_equal(loaded.obs_rms.mean, final["obs_rms"]["mean"])
                    np.testing.assert_array_equal(loaded.obs_rms.var, final["obs_rms"]["var"])
                    self.assertEqual(loaded.obs_rms.count, final["obs_rms"]["count"])
                if norm_reward:
                    np.testing.assert_array_equal(loaded.return_rms.var, final["return_rms"]["var"])

                # 2. Resume continues: 2 more updates to reach 4
                resumed = self.make_trainer(os.path.join(tmp, "resumed"))
                resumed_info = resumed.train(self.make_train_config(4*STEP, norm_obs, norm_reward, 2), ppo_config, resume_from=mid_path)
                self.assertTrue(resumed_info["resumed"])
                self.assertEqual(resumed_info["start_update"], 2)
                resumed_ckpt = torch.load(os.path.join(ckpt_dir("resumed"), "latest.pt"), weights_only=False)
                self.assertEqual(resumed_ckpt["update"], 4)
                self.assertEqual(resumed_ckpt["global_step"], 4*STEP)
                self.assertEqual(resumed.opt.param_groups[0]["lr"], 0.0)           # schedule finished, not restarted
                self.assertFalse(os.path.exists(os.path.join(ckpt_dir("resumed"), f"checkpoint_{2*STEP:09d}.pt")))  # didn't redo update 2
                self.assertFalse(torch.equal(resumed.actor.state_dict()["net.0.weight"], mid["actor"]["net.0.weight"]))  # it trained
        
    def test_resume_config_check(self):
        STEP = 64 * 2
        ppo_config = ppo.PPOTrainerPPOUpdateConfig()
        with tempfile.TemporaryDirectory() as tmp:
            base = self.make_trainer(os.path.join(tmp, "base"))
            base.train(self.make_train_config(2*STEP, True, False, None), ppo_config)
            path = os.path.join(tmp, "base", "checkpoints", "CartPole-v1", "seed_0", "latest.pt")

            def resume(train_config, ppo=ppo_config):
                # Fresh run_dir each time; run_dir is in RESUME_IGNORE so this mustn't trigger anything
                trainer = self.make_trainer(tempfile.mkdtemp(dir=tmp))
                trainer.train(train_config, ppo, resume_from=path)

            # Matching config (apart from ignored fields): no error and no ResumeConfigWarning
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                resume(self.make_train_config(2*STEP, True, False, 1))   # checkpoint_every differs, but is ignored
            self.assertFalse([w for w in caught if issubclass(w.category, ppo.ResumeConfigWarning)])

            # Incompatible fields: refuse to resume, and name the field in the error
            rollout_changed = self.make_train_config(2*STEP, True, False, None)
            rollout_changed.rollout_steps = 32
            incompatible = {
                "total_steps":            self.make_train_config(4*STEP, True, False, None),
                "normalise_observations": self.make_train_config(2*STEP, False, False, None),
                "normalise_reward":       self.make_train_config(2*STEP, True, True, None),
                "rollout_steps":          rollout_changed,
            }
            for field, train_config in incompatible.items():
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, field):
                        resume(train_config)

            # Other changes: resume, but warn and name the field
            ppo_changed = ppo.PPOTrainerPPOUpdateConfig()
            ppo_changed.ent_coef = 0.0
            with self.assertWarnsRegex(ppo.ResumeConfigWarning, "ent_coef"):
                resume(self.make_train_config(2*STEP, True, False, None), ppo=ppo_changed)

if __name__ == "__main__":
    unittest.main()
