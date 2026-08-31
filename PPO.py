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
    obs:        List = field(default_factory=list)
    actions:    List = field(default_factory=list)
    logprobs:   List = field(default_factory=list)  # log π_old(a_t | s_t)
    rewards:    List = field(default_factory=list)
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

def make_envs(env_id: str, num_envs:int=1, seed: int=0):
    """Makes a vector of environments"""
    def make_one(i):
        def _thunk():
            env = gym.make(env_id)
            env.action_space.seed(seed)
            return env
        return _thunk
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

def collect_rollout(env,
    actor: Actor,
    critic: Critic,
    steps: int, device,
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

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                last_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95):
    """Generalized Advantage Estimation.

    TODO: implement.
      For t = T-1, T-2, ..., 0:
        next_value    = values[t+1]  if t+1 < T else last_value
        next_nonterm  = 1.0 - dones[t]
        delta_t       = rewards[t] + gamma * next_value * next_nonterm - values[t]
        A_t           = delta_t + gamma * lam * next_nonterm * A_{t+1}    (with A_T = 0)

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

def train_vec(update_fn: Callable,
    env_id: str,
    total_steps: int = 50_000,
    rollout_steps: int = 512,
    seed: int = 0,
    verbose: bool = True,
    lr=2.5e-4,
    adam_eps=1e-8,
    discountReturns=False,
    gamma=0.99,
    num_envs=4,
    device=DEFAULT_DEVICE,
    **update_kwargs):
    """Generic PPO-style training loop.
    
    Vector implemetation, note the only differences to train are the make_env_vec and collect_rollout_vec calls

    update_fn(rollout_dict, actor, critic, actor_opt, critic_opt, **update_kwargs) -> logs dict
    """
    env = make_envs(env_id, num_envs=num_envs, seed=seed)
    obs_dim = env.single_observation_space.shape[0]
    n_actions = env.single_action_space.n

    actor = Actor(obs_dim, n_actions).to(device)
    critic = Critic(obs_dim).to(device)
    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=lr, eps=adam_eps)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr, eps=adam_eps)

    all_returns: List[float] = []
    all_lengths: List[int] = []
    recent = deque(maxlen=20)
    n_updates = total_steps // rollout_steps
    
    actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(actor_opt, start_factor=1.0, end_factor=0.0, total_iters=n_updates)
    critic_lr_scheduler = torch.optim.lr_scheduler.LinearLR(critic_opt, start_factor=1.0, end_factor=0.0, total_iters=n_updates)
    
    return_rms = None
    discounted_return = np.zeros(num_envs)
    if discountReturns:
        return_rms = RunningMeanStd()

    for update in range(n_updates):
        rollout = collect_rollout(env, actor, critic, rollout_steps, device, return_rms, discounted_return, gamma)
        for ep_ret, ep_len in rollout.episode_stats:
            all_returns.append(ep_ret)
            all_lengths.append(ep_len)
            recent.append(ep_ret)

        logs = update_fn(rollout.to_tensors(device), actor, critic,
                         actor_opt, critic_opt, **update_kwargs)

        if verbose and (update % max(1, n_updates // 20) == 0 or update == n_updates - 1):
            steps_seen = (update + 1) * rollout_steps
            recent_mean = np.mean(recent) if recent else float('nan')
            log_str = " ".join(f"{k}={v:.3f}" for k, v in (logs or {}).items())
            print(f"[{update+1:>4}/{n_updates},lr={actor_lr_scheduler.get_last_lr()[0]:.2e}] steps={steps_seen:>6}  "
                  f"recent_ret={recent_mean:6.1f}  {log_str}")
          
        actor_lr_scheduler.step()
        critic_lr_scheduler.step()

    env.close()
    return {'returns': all_returns, 'lengths': all_lengths}
    
def ppo_update_polished(batch, actor, critic, actor_opt, critic_opt, **cfg):
    """Phase 5: PPO with the trick checklist.

    TODO: extend ppo_update with the tricks above, one at a time.
    Suggested config keys:
        clip_eps, epochs, minibatch_size, vf_coef, ent_coef,
        clip_vloss (bool), max_grad_norm, norm_adv (bool), target_kl (float or None)
    Return diagnostics: approx_kl, clip_frac, explained_var, entropy.
    """
    gamma: float = cfg.get("gamma", 0.99)
    lam: float = cfg.get("lam", 0.95)
    clip_eps: float = cfg.get("clip_eps", 0.2)
    epochs: int = cfg.get("epochs", 4)
    minibatch_size: int = cfg.get("minibatch_size", 64)
    vf_coef: float = cfg.get("vf_coef", 0.5)
    ent_coef: float = cfg.get("ent_coef", 0.0)
    clip_vloss: bool = cfg.get("clip_vloss", True)
    norm_adv: bool = cfg.get("norm_adv", True)
    target_kl: float = cfg.get("target_kl", None)
    max_grad_norm: float = cfg.get("max_grad_norm", 0.5)
    
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
            
            actor_opt.zero_grad()
            critic_opt.zero_grad()
            
            loss.backward()
            
            # Cap how far a single update can move the parameters, no matter how large the gradient. A safety net
            # for when normalisation isn't enough
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            
            actor_opt.step()
            critic_opt.step()
            
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

def main(args):
    if len(args) < 2:
        print("Require environment")
        exit()
        
    env_name = args[1]
    print("Environment: ", env_name)
    
    seed = 0
    total_steps = 50000 #200_000
    start_time = timeit.default_timer()
    num_envs = 1
    rollout_steps = 512 // num_envs
    device = DEFAULT_DEVICE
    
    run = train_vec(ppo_update_polished, env_name, total_steps=total_steps, seed=seed,
        clip_eps=0.2, epochs=4, minibatch_size=64,
        norm_adv=True, clip_vloss=True, max_grad_norm=0.5,
        ent_coef=0.01, target_kl=0.015,
        num_envs=num_envs, rollout_steps=rollout_steps,
        lr=2.5e-4, adam_eps=1e-5, discountReturns=True, device=device)['returns']
    elapsed = timeit.default_timer() - start_time
    print("Elapsed: ", elapsed)

if __name__ == "__main__":
    main(sys.argv)
