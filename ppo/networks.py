import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

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

class SimpleActor(nn.Module):   
    def __init__(self, obs_space: int, action_space: int, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_space.shape[0]
        self.n_actions = action_space.n
        self.net = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            # Policy head: small init so initial logits are ~uniform
            layer_init(nn.Linear(hidden, self.n_actions), std=0.01),
        )
    def forward(self, x):
        return self.net(x)  # logits
    def dist(self, x):
        return Categorical(logits=self.forward(x))

class SimpleCritic(nn.Module):
    def __init__(self, obs_space: int, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_space.shape[0]
        self.net = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            ppo.layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            ppo.layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            ppo.layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            ppo.layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.cnn(x)

class VisionActor(nn.Module):
    def __init__(self, obs_space: int, action_space: int, cnn: CNNEncoder, hidden: int = 64):
        super().__init__()

        self.obs_dim = obs_space.shape[0]
        self.n_actions = action_space.n
        print(f"Obs dim: {self.obs_dim}, Actions Dim: {self.n_actions}")

        self.cnn = cnn

        self.net = nn.Sequential(
            ppo.layer_init(nn.Linear(512, hidden)), nn.Tanh(),
            ppo.layer_init(nn.Linear(hidden, self.n_actions), std=0.01),
        )

    def forward(self, x):
        features = self.cnn(x)
        logits = self.net(features)  # logits
        return logits
    def dist(self, x):
        return Categorical(logits=self.forward(x))

class VisionCritic(nn.Module):
    def __init__(self, obs_space: int, cnn: CNNEncoder, hidden: int = 64):
        super().__init__()
        self.cnn = cnn
        self.net = nn.Sequential(
            ppo.layer_init(nn.Linear(512, hidden)), nn.Tanh(),
            ppo.layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, x):
        features = self.cnn(x)
        return self.net(features).squeeze(-1)