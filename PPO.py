import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Optional
from collections import deque
import sys
import timeit

DEFAULT_DEVICE="cpu"

@dataclass
class Rollout:
    """Represents a rollout of several trajectories for one or more environments.
    
    Usually used as a fixed length rollout, that for one environment would look like:
    [rollout 1, x steps][rollout 2, y steps]...[rollout n, z steps truncated]
    And last_value is set to the value of the next state after the final step.
    
    We store 'dones' so we know where each trajectory ended.
    """
    obs:        List = field(default_factory=list)  # Observations
    actions:    List = field(default_factory=list)  # Actions
    logprobs:   List = field(default_factory=list)  # Log probabilities of actions: log π_old(a_t | s_t)
    rewards:    List = field(default_factory=list)  # Rewards
    dones:      List = field(default_factory=list)  # 1.0 if episode ended after step t
    values:     List = field(default_factory=list)  # V_φ(s_t)
    last_value: float = 0.0                         # V_φ(s_T)

    def add(self, obs, action, logprob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def to_tensors(self, device):
        return {
            'obs':        torch.as_tensor(np.asarray(self.obs),      dtype=torch.float32, device=device),
            'actions':    torch.as_tensor(np.asarray(self.actions),  dtype=torch.long,    device=device),
            'logprobs':   torch.as_tensor(np.asarray(self.logprobs), dtype=torch.float32, device=device),
            'rewards':    torch.as_tensor(np.asarray(self.rewards),  dtype=torch.float32, device=device),
            'dones':      torch.as_tensor(np.asarray(self.dones),    dtype=torch.float32, device=device),
            'values':     torch.as_tensor(np.asarray(self.values),   dtype=torch.float32, device=device),
            'last_value': torch.as_tensor(np.asarray(self.last_value),  dtype=torch.float32, device=device),
        }

    def __len__(self):
        return len(self.rewards)

class RunningMeanStd:
    """Used to normalise rewards.
    Note: gym has a wrapper that does this, gym.Wrappers.NormalizeReward
    but we do it manually to show how it's done
    """
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4  # small epsilon to avoid div-by-zero
        
    def update(self, x):
        # Welford's online algorithm — numerically stable running variance
        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item()
        batch_count = x.numel()
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

def make_envs(env_id: str, num_envs:int=1, async_envs:bool = False, seed: int=0, make_env_callback=None):
    """Makes a vector of environments"""
    def make_one(i):
        def _thunk():
            env = gym.make(env_id)
            env.action_space.seed(seed)
            if make_env_callback != None:
                env = make_env_callback(env)
            return env
        return _thunk
    if async_envs:
        return gym.vector.AsyncVectorEnv([make_one(i) for i in range(num_envs)])
    else:
        return gym.vector.SyncVectorEnv([make_one(i) for i in range(num_envs)])

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0):
    """Initialises a layer"""
    # Orthogonal initialization fills a weight matrix with random values, then projects the matrix onto the nearest
    # orthogonal matrix. As orthogonal matrices preserve norms. When you pass a vector x through Wx, ||Wx|| = ||x||
    # if W is orthogonal.
    # Stack many such layers and activations neither explode nor vanish at initialization — which is a very different
    # starting point from the default PyTorch init (Kaiming uniform for nn.Linear), where the scale of pre-activations
    # can drift meaningfully through depth
    
    # The std (gain) parameter scales the orthogonal matrix
    # - std=sqrt(2) for hidden layers: this is "He initialization" adapted for orthogonal — it's the gain that preserves
    #   variance (not just norm) through a ReLU-like nonlinearity, which is a stronger property than norm preservation alone.
    #   Signals stay well-scaled through the network at init
    # - std=0.01 for the policy head: this is the important one. When the final Linear layer has tiny weights, its output
    #   logits are near zero for any input, which means softmax(logits) is nearly uniform — every action gets roughly equal
    #   probability. This gives the policy maximum entropy at initialization, so it explores broadly for the first few updates
    #   before the policy has any real information to concentrate on. Without this, random init might produce logits that already
    #   strongly favor one action, and you spend the early episodes "un-learning" that arbitrary bias. This is arguably the single
    #   most impactful trick for stable early training.
    # - std=1.0 for the value head: preserves the natural scale, so V(s) starts near zero rather than at some arbitrary offset.
    #   This matters because the critic's target (returns) is often around zero early on, and starting with V ≈ 0 means the critic
    #   doesn't have to first "unwind" a large initial bias before learning the actual mapping
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, val=bias)
    return layer

class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            # Policy head: small init so initial logits are ~uniform
            layer_init(nn.Linear(hidden, n_actions), std=0.01),
        )
    def forward(self, x):
        return self.net(x)  # logits
    def dist(self, x):
        return Categorical(logits=self.forward(x))

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def collect_rollout(
    env,
    actor: Actor,
    critic: Critic,
    steps: int,
    device,
    return_rms: RunningMeanStd = None,
    discounted_return = 0.0,
    gamma = 1.0) -> Rollout:
    """Collects a rollout from an environment"""
    N = env.num_envs
    r = Rollout()
    obs, _ = env.reset()  # shape (N, obs_dim)
    ep_returns_running = np.zeros(N)
    ep_lens_running = np.zeros(N, dtype=int)
    ep_stats = []

    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)  # (N, obs_dim)
        with torch.no_grad():
            dist = actor.dist(obs_t)
            action = dist.sample()                # (N,)
            logprob = dist.log_prob(action)       # (N,)
            value = critic(obs_t)                 # (N,)

        next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)  # (N,)
        raw_reward = reward.copy()                    # (N,) — save before normalization

        # If we have a RunningMeanStd, normalise the rewards
        if return_rms is not None:
            discounted_return = discounted_return * gamma + reward
            return_rms.update(torch.as_tensor(discounted_return))
            reward = reward / (np.sqrt(return_rms.var) + 1e-8)
            discounted_return[done] = 0.0             # reset accumulator per env that ended

        # Store the whole N-wide slice as one row
        r.add(obs, action.cpu().numpy(), logprob.cpu().numpy(), reward, done.astype(np.float32), value.cpu().numpy())

        ep_returns_running += raw_reward
        ep_lens_running += 1
        for i in range(N):
            if done[i]:
                ep_stats.append((ep_returns_running[i], ep_lens_running[i]))
                ep_returns_running[i] = 0.0
                ep_lens_running[i] = 0

        obs = next_obs  # auto-reset already handled by the vec env

    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        r.last_value = critic(obs_t).cpu().numpy()   # (N,) now, not scalar

    r.episode_stats = ep_stats
    return r

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95):
    """Generalized Advantage Estimation.

    Args:
        rewards:    (T,) float
        values:     (T,) float
        dones:      (T,) float
        last_value: scalar tensor
        gamma:      discount
        lam:        GAE lambda
    Returns:
        advantages: (T,) float
        returns:    (T,) float, = advantages + values (target for the critic)
    """
    T = rewards.shape[0]
    V_t_1 = last_value
    A_t_1 = torch.tensor(0.0)
    
    advantages = torch.zeros_like(rewards)    # (N,)
    returns = torch.zeros_like(rewards)
    
    for t in reversed(range(T)):
        # The temporal difference (TD) error \(\delta _{t}\) is an unbiased
        # sample of the advantage function \(A(s, a)\) because its conditional
        # expectation given the current state and action is exactly equal to the advantage.
        delta_t = rewards[t] + gamma * V_t_1 * (1.0-dones[t]) - values[t]
        # The recursive form of GAE
        A_t = delta_t + gamma * lam * (1.0-dones[t]) * A_t_1
        
        advantages[t] = A_t
        returns[t] = A_t + values[t]
        
        V_t_1 = values[t]
        A_t_1 = A_t
        
    return advantages, returns

@dataclass
class PPOTrainerConfig:
    """Basic parameters for the PPOTrainer class
    """
    num_envs = 1
    async_envs = False
    seed = 0
    device = DEFAULT_DEVICE
    make_env_callback = None    # Called for each env after creation, for additional gym wrappers, etc

@dataclass
class PPOTrainerTrainConfig:
    """Common RL training paramters
    """
    total_steps: int = 50_000
    rollout_steps: int = 512
    verbose: bool = True
    lr = 2.5e-4
    adam_eps = 1e-8
    discountReturns = True
    gamma = 0.99

@dataclass
class PPOTrainerPPOUpdateConfig:
    """PPO focused training parameters
    """
    clip_eps=0.2
    epochs=4
    minibatch_size=64
    norm_adv=True
    clip_vloss=True
    max_grad_norm=0.5
    ent_coef=0.0
    target_kl=None
    lam: float = 0.95    
    vf_coef: float = 0.5

def ppo_update(batch, actor, critic, opt, trainConfig: PPOTrainerTrainConfig, ppoConfig: PPOTrainerPPOUpdateConfig):
    """PPO Update
    """
    
    # Unpack the config parameters we need
    gamma: float = trainConfig.gamma
    lam: float = ppoConfig.lam
    clip_eps: float = ppoConfig.clip_eps
    epochs: int = ppoConfig.epochs
    minibatch_size: int = ppoConfig.minibatch_size
    vf_coef: float = ppoConfig.vf_coef
    ent_coef: float = ppoConfig.ent_coef
    clip_vloss: bool = ppoConfig.clip_vloss
    norm_adv: bool = ppoConfig.norm_adv
    target_kl: float = ppoConfig.target_kl
    max_grad_norm: float = ppoConfig.max_grad_norm
    
    # Compute GAE advantages and returns
    advantages, returns = compute_gae(batch["rewards"],
        batch["values"],
        batch["dones"],
        batch["last_value"],
        gamma,
        lam)
    
    # Flatten if we got a vec batch. Must be AFTER GAE — reshape interleaves envs.
    if batch["obs"].dim() == 3:
        T, N = batch["obs"].shape[:2]
        batch = {
            "obs":      batch["obs"].reshape(T*N, -1),
            "actions":  batch["actions"].reshape(T*N),
            "logprobs": batch["logprobs"].reshape(T*N),
            "values":   batch["values"].reshape(T*N),
            "rewards":  batch["rewards"].reshape(T*N),
            "last_value": batch["last_value"],  # (N,), not used past this point
        }
        advantages = advantages.reshape(T*N)
        returns    = returns.reshape(T*N)
    
    clip_frac = []
    policy_losses = []
    value_losses = []
    epochs_used = 0
    update_kls = []
        
    for epoch in range(epochs):
        epoch_kls = []
        permutation = torch.randperm(batch["obs"].size(0), device=batch["obs"].device)
        
        for i in range(0, batch["obs"].size(0), minibatch_size):
            # Create the minibatch
            indices = permutation[i:i+minibatch_size]
            minibatch = {
                "obs":batch["obs"][indices],
                "actions":batch["actions"][indices],
                "logprobs":batch["logprobs"][indices],
                "rewards":batch["rewards"][indices],
                "values":batch["values"][indices],
            }
            
            minibatch_advantages = advantages[indices]
            
            # normalise the advantages for the minibatch
            if norm_adv:
                advantages_norm = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)
                minibatch_advantages = advantages_norm
            
            # Update policies
            dist = actor.dist(minibatch['obs'])
            logprobs_new = dist.log_prob(minibatch["actions"])
            ratio = (logprobs_new - minibatch["logprobs"]).exp()
            surr1 = ratio * minibatch_advantages
            surr2 = ratio.clamp(1-clip_eps, 1+clip_eps) * minibatch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            v_new = critic(minibatch["obs"])
            v_old = minibatch["values"]
            loss_unclipped = (v_new - returns[indices]) ** 2
            if clip_vloss:
                # clipped value loss prevents overshoots in the direction the critic already moved, while still permitting 
                # corrections back toward v_old.
                # eg. v_old=5, v_new=10, returns=15 => clipped loss dominates and cannot go beyond trust region
                #     v_old=5, v_new=10, returns=0 => natural loss is back towards v_old, unclipped dominates and update is allowed
                v_clipped = v_old + torch.clamp(v_new - v_old, -clip_eps, clip_eps)
                loss_clipped = (v_clipped - returns[indices]) ** 2
                # 0.5 is a convention that cancels out the 2 from the derivative of the square
                value_loss = 0.5 * torch.max(loss_unclipped, loss_clipped).mean()
            else:
                value_loss = 0.5 * loss_unclipped.mean()
            
            entropy = dist.entropy().mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            
            opt.zero_grad()
            
            loss.backward()
            
            # Cap how far a single update can move the parameters, no matter how large the gradient. A safety net
            # for when normalisation isn't enough
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            
            opt.step()
            
            with torch.no_grad():
                log_ratio = logprobs_new - minibatch["logprobs"]
                approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean().item()
                epoch_kls.append(approx_kl)
                clip_frac.append(((ratio - 1.0).abs() > clip_eps).float().mean().item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        
        epochs_used += 1
        # Early out if we have a target kl and we have exceeded it
        # You can think of target_kl as a data-freshness check: "is the data I'm training on still representative of my current policy?
        # If not, throw it out and go collect fresh data."
        update_kls.append(np.mean(epoch_kls))
        if target_kl is not None and np.mean(epoch_kls) > target_kl:
            break
            
    with torch.no_grad():
        var_returns = returns.var()
        if var_returns > 0:
            explained_var = 1.0 - (returns - batch["values"]).var() / var_returns
        else:
            explained_var = torch.tensor(float('nan'))
    
    return {
        'actor_loss':float(np.mean(policy_losses)),
        'value_loss':float(np.mean(value_losses)),
        'approx_kl':float(np.mean(update_kls)),
        'clip_frac':float(np.mean(clip_frac)),
        'explained_variance':explained_var.item(),
        'epochs_used':epochs_used}


class PPOTrainer:
    def __init__(self, env_id: str, config: PPOTrainerConfig):
        
        # Get the PPO setup from the passed in config
        self.num_envs = config.num_envs
        self.async_envs = config.async_envs
        self.seed = config.seed
        self.device = config.device
        self.make_env_callback = config.make_env_callback
        
        # Create environment and store obs and action spaces
        self.env = make_envs(env_id, num_envs=self.num_envs, async_envs=self.async_envs, seed=self.seed, make_env_callback=self.make_env_callback)
        self.obs_dim = self.env.single_observation_space.shape[0]
        self.n_actions = self.env.single_action_space.n

        # Create the actor/critic architectures
        self.actor = Actor(self.obs_dim, self.n_actions).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)
    
    def train(self,
        trainConfig: PPOTrainerTrainConfig,
        ppoConfig: PPOTrainerPPOUpdateConfig):
        """Generic PPO-style training loop
        """
        
        # Get the training config values
        total_steps: int = trainConfig.total_steps
        rollout_steps: int = trainConfig.rollout_steps
        verbose: bool = trainConfig.verbose
        lr = trainConfig.lr
        adam_eps = trainConfig.adam_eps
        discountReturns: bool = trainConfig.discountReturns
        gamma = trainConfig.gamma,
        
        # update_fn(rollout_dict, actor, critic, opt, **update_kwargs) -> logs dict
        update_fn: Callable = ppo_update

        # Create the actor/critic optimisers
        total_params = list(set(self.actor.parameters()) | set(self.critic.parameters()))
        opt  = torch.optim.Adam(total_params,  lr=lr, eps=adam_eps)
        
        all_returns: List[float] = []
        all_lengths: List[int] = []
        recent = deque(maxlen=20)
        n_updates = total_steps // rollout_steps
    
        # Create learning rate schedulers for the optimisers
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=n_updates)
    
        # Setup discounted returns and running mean/std for normalisation
        # TODO: I think we should be normalising even if discounted_returns isn't set, check this
        return_rms = None
        discounted_return = np.zeros(self.num_envs)
        if discountReturns:
            return_rms = RunningMeanStd()

        for update in range(n_updates):
            # Collect a fixed length rollout (number of trajectories, with the last one truncated), for each environment
            rollout = collect_rollout(self.env, self.actor, self.critic, rollout_steps, self.device, return_rms, discounted_return, gamma)
            for ep_ret, ep_len in rollout.episode_stats:
                all_returns.append(ep_ret)
                all_lengths.append(ep_len)
                recent.append(ep_ret)

            # Do the PPO Update
            logs = update_fn(rollout.to_tensors(self.device), self.actor, self.critic,
                             opt, trainConfig, ppoConfig)

            # Log stats
            if verbose and (update % max(1, n_updates // 20) == 0 or update == n_updates - 1):
                steps_seen = (update + 1) * rollout_steps
                recent_mean = np.mean(recent) if recent else float('nan')
                log_str = " ".join(f"{k}={v:.3f}" for k, v in (logs or {}).items())
                print(f"[{update+1:>4}/{n_updates},lr={lr_scheduler.get_last_lr()[0]:.2e}] steps={steps_seen:>6}  "
                      f"recent_ret={recent_mean:6.1f}  {log_str}")
          
            # Update learning rate schedulers
            lr_scheduler.step()

        self.env.close()
        return {'returns': all_returns, 'lengths': all_lengths}
        

def main(args):
    if len(args) < 2:
        print("Usage: PPO.py <environment name>")
        exit()
        
    env_name = args[1]
    print("Environment: ", env_name)
    
    ppoTrainerConfig = PPOTrainerConfig()
    ppoTrainerConfig.num_envs = 1
    ppoTrainerConfig.seed = 0
    ppoTrainerConfig.device = DEFAULT_DEVICE
    
    trainer = PPOTrainer(env_name, ppoTrainerConfig)
    
    trainConfig = PPOTrainerTrainConfig()
    trainConfig.total_steps = 50000 #200_000
    trainConfig.rollout_steps = 512 # // ppoTrainerConfig.num_envs
    trainConfig.lr = 2.5e-4
    trainConfig.adam_eps = 1e-5
    trainConfig.discountReturns = True
    
    ppoConfig = PPOTrainerPPOUpdateConfig()
    ppoConfig.clip_eps=0.2
    ppoConfig.epochs=4
    ppoConfig.minibatch_size=64
    ppoConfig.norm_adv=True
    ppoConfig.clip_vloss=True
    ppoConfig.max_grad_norm=0.5
    ppoConfig.ent_coef=0.01
    ppoConfig.target_kl=0.015
    ppoConfig.lam: float = 0.95    
    ppoConfig.vf_coef: float = 0.5
    
    start_time = timeit.default_timer()
    
    run = trainer.train(trainConfig, ppoConfig)['returns']
    elapsed = timeit.default_timer() - start_time
    print("Elapsed: ", elapsed)

if __name__ == "__main__":
    main(sys.argv)
