from models.DQN import DQNModel
from models.mask import MaskModel
from models.transition import DeterministicTransitionModel
from models.reward import RewardModel
import torch
import torch.nn as nn
from collections import namedtuple, deque
import random
import numpy as np
from sklearn.cluster import DBSCAN
import torch.nn.functional as F
import os

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminate'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# =============================================================================
# Policy Networks
# =============================================================================

class DiscreteSACPolicy(nn.Module):
    """Actor network for a single discrete action space."""
    def __init__(self, obs_n, act_n, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_n, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_n),
        )

    def forward(self, state):
        return self.net(state)

    def get_action_probs(self, state, mask=None):
        logits = self.forward(state)
        if mask is not None:
            logits = logits + (mask.clamp(min=1e-8).log())
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp(min=1e-8))
        return probs, log_probs


class MultiDiscreteSACPolicy(nn.Module):
    """Actor network for multi-discrete action spaces.

    Each action dimension gets its own softmax head branching from a shared
    feature trunk.

    Args:
        obs_n:    observation dimension
        act_dims: list/tuple of ints, e.g. [3, 4, 5] for MultiDiscrete([3,4,5])
        hidden:   hidden layer width
    """
    def __init__(self, obs_n, act_dims, hidden=128):
        super().__init__()
        self.act_dims = act_dims
        self.num_heads = len(act_dims)

        self.trunk = nn.Sequential(
            nn.Linear(obs_n, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # One logit head per action dimension
        self.heads = nn.ModuleList([
            nn.Linear(hidden, d) for d in act_dims
        ])

    def forward(self, state):
        """Returns a list of logit tensors, one per action dimension."""
        features = self.trunk(state)
        return [head(features) for head in self.heads]

    def get_action_probs(self, state, masks=None):
        """
        Args:
            state:  (batch, obs_n)
            masks:  list of (batch, act_dims[i]) tensors, or None

        Returns:
            probs_list:     list of (batch, act_dims[i])
            log_probs_list: list of (batch, act_dims[i])
        """
        logits_list = self.forward(state)
        probs_list, log_probs_list = [], []
        for i, logits in enumerate(logits_list):
            if masks is not None and masks[i] is not None:
                logits = logits + masks[i].clamp(min=1e-8).log()
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs.clamp(min=1e-8))
            probs_list.append(probs)
            log_probs_list.append(log_probs)
        return probs_list, log_probs_list


# =============================================================================
# Q-Network for multi-discrete (factorised per-dimension)
# =============================================================================

class MultiDiscreteQNetwork(nn.Module):
    """Factorised Q-network: one Q-head per action dimension.

    Each head outputs Q(s, a_i) for all actions in dimension i.
    Total Q ≈ sum of per-dimension Q-values (linear decomposition).
    """
    def __init__(self, obs_n, act_dims, hidden=128):
        super().__init__()
        self.act_dims = act_dims
        self.trunk = nn.Sequential(
            nn.Linear(obs_n, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, d) for d in act_dims
        ])

    def forward(self, state):
        """Returns list of (batch, act_dims[i]) Q-value tensors."""
        features = self.trunk(state)
        return [head(features) for head in self.heads]


# =============================================================================
# Base SAC — single discrete
# =============================================================================

class SAC:
    """Discrete Soft Actor-Critic (single discrete action space)."""
    def __init__(
            self,
            obs_n,
            act_n,
            device,
            policy_lr=3e-4,
            q_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            batch_size=256,
            replay_memory_size=2000,
            trans_freq=50,
            init_alpha=0.2,
            learnable_alpha=True,
    ):
        self.obs_n = obs_n
        self.act_n = act_n
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.trans_freq = trans_freq
        self.update_steps = 0

        # Actor
        self.policy = DiscreteSACPolicy(obs_n, act_n).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

        # Twin Q-networks
        self.q1 = DQNModel(obs_n, act_n).to(device)
        self.q2 = DQNModel(obs_n, act_n).to(device)
        self.q1_target = DQNModel(obs_n, act_n).to(device)
        self.q2_target = DQNModel(obs_n, act_n).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=q_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=q_lr)

        # Entropy temperature
        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            self.target_entropy = -np.log(1.0 / act_n) * 0.98
            self.log_alpha = torch.tensor(np.log(init_alpha), device=device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(init_alpha), device=device)

        self.replay_memory = ReplayMemory(replay_memory_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, state, action=None, mask=None):
        probs, log_probs = self.policy.get_action_probs(state, mask=mask)
        if action is None:
            action = torch.multinomial(probs, 1).squeeze(-1)
        return action, probs, mask

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def update(self, load_model):
        trans_loss, reward_loss, mask_loss = None, None, None
        if len(self.replay_memory) < self.batch_size:
            return trans_loss, reward_loss, mask_loss

        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        t = torch.tensor(batch.terminate, device=self.device, dtype=torch.bool).detach()
        s_ = torch.cat(batch.next_state).detach()
        s = torch.cat(batch.state).detach()
        a = torch.cat(batch.action).detach()
        r = torch.cat(batch.reward).detach()

        self.update_policy(s, a, r, s_, t)

        if not load_model and self.update_steps % self.trans_freq == 0:
            self.update_steps = 0
            trans_loss, reward_loss = self.update_transition_reward(s, a, r.unsqueeze(1), s_)
            mask_loss = self.update_mask(s)

        return trans_loss, reward_loss, mask_loss

    def update_policy(self, s, a, r, s_, t):
        # ---- Q targets ----
        with torch.no_grad():
            next_probs, next_log_probs = self.policy.get_action_probs(s_)
            q1_next = self.q1_target(s_)
            q2_next = self.q2_target(s_)
            min_q_next = torch.min(q1_next, q2_next)
            next_v = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(-1)
            target_q = r + self.gamma * next_v * t

        a_idx = a.unsqueeze(-1) if a.dim() == 1 else a
        q1_vals = self.q1(s).gather(1, a_idx).squeeze(-1)
        q2_vals = self.q2(s).gather(1, a_idx).squeeze(-1)

        q1_loss = F.mse_loss(q1_vals, target_q)
        q2_loss = F.mse_loss(q2_vals, target_q)

        self.q1_optimizer.zero_grad(); q1_loss.backward()
        torch.nn.utils.clip_grad_value_(self.q1.parameters(), 100)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad(); q2_loss.backward()
        torch.nn.utils.clip_grad_value_(self.q2.parameters(), 100)
        self.q2_optimizer.step()

        # ---- Policy ----
        probs, log_probs = self.policy.get_action_probs(s)
        min_q_pi = torch.min(self.q1(s), self.q2(s))
        policy_loss = (probs * (self.alpha.detach() * log_probs - min_q_pi)).sum(-1).mean()

        self.policy_optimizer.zero_grad(); policy_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.policy_optimizer.step()

        # ---- Alpha ----
        if self.learnable_alpha:
            entropy = -(probs.detach() * log_probs.detach()).sum(-1).mean()
            alpha_loss = self.log_alpha * (entropy - self.target_entropy)
            self.alpha_optimizer.zero_grad(); alpha_loss.backward()
            self.alpha_optimizer.step()

        self.update_steps += 1
        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)

    def update_transition_reward(self, s, a, r, s_):
        return None, None

    def update_mask(self, s):
        return None

    def save_model(self, save_path):
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
        }, os.path.join(save_path, 'sac_model.pth'))

    def load_model(self, load_path):
        pass


# =============================================================================
# Base SAC — multi-discrete
# =============================================================================

class MultiDiscreteSAC(SAC):
    """Discrete SAC for MultiDiscrete action spaces.

    Uses factorised per-dimension policy heads and Q-heads.
    Actions are stored/sampled as (batch, num_dims) integer tensors.

    Args:
        obs_n:    observation dimension
        act_dims: list of ints, e.g. [3, 4, 5]
    """
    def __init__(
            self,
            obs_n,
            act_dims,
            device,
            policy_lr=3e-4,
            q_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            batch_size=256,
            replay_memory_size=2000,
            trans_freq=50,
            init_alpha=0.2,
            learnable_alpha=True,
    ):
        # Skip SAC.__init__ — we override all network creation
        self.obs_n = obs_n
        self.act_dims = list(act_dims)
        self.num_action_dims = len(self.act_dims)
        self.act_n = int(np.prod(self.act_dims))  # total combos (for reference)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.trans_freq = trans_freq
        self.update_steps = 0

        # Factorised actor
        self.policy = MultiDiscreteSACPolicy(obs_n, self.act_dims).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

        # Factorised twin Q-networks (one Q-head per dimension)
        self.q1 = MultiDiscreteQNetwork(obs_n, self.act_dims).to(device)
        self.q2 = MultiDiscreteQNetwork(obs_n, self.act_dims).to(device)
        self.q1_target = MultiDiscreteQNetwork(obs_n, self.act_dims).to(device)
        self.q2_target = MultiDiscreteQNetwork(obs_n, self.act_dims).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=q_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=q_lr)

        # Per-dimension entropy temperature
        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            self.target_entropies = [
                -np.log(1.0 / d) * 0.98 for d in self.act_dims
            ]
            self.log_alphas = [
                torch.tensor(np.log(init_alpha), device=device, requires_grad=True)
                for _ in self.act_dims
            ]
            self.alpha_optimizers = [
                torch.optim.Adam([la], lr=alpha_lr) for la in self.log_alphas
            ]
        else:
            self.log_alphas = [
                torch.tensor(np.log(init_alpha), device=device)
                for _ in self.act_dims
            ]

        self.replay_memory = ReplayMemory(replay_memory_size)

    def alphas(self):
        """Return list of alpha scalars."""
        return [la.exp() for la in self.log_alphas]

    def get_action(self, state, action=None, masks=None):
        """
        Args:
            state:  (batch, obs_n)
            action: (batch, num_dims) or None
            masks:  list of (batch, act_dims[i]) or None

        Returns:
            action: (batch, num_dims)  LongTensor
            probs_list: list of (batch, act_dims[i])
            masks
        """
        probs_list, _ = self.policy.get_action_probs(state, masks=masks)
        if action is None:
            action = torch.stack(
                [torch.multinomial(p, 1).squeeze(-1) for p in probs_list], dim=-1
            )
        return action, probs_list, masks

    def update_policy(self, s, a, r, s_, t):
        """
        a is expected as (batch, num_dims) LongTensor.
        """
        alphas = self.alphas()

        # ---- Q targets ----
        with torch.no_grad():
            next_probs_list, next_log_probs_list = self.policy.get_action_probs(s_)
            q1_next_list = self.q1_target(s_)
            q2_next_list = self.q2_target(s_)

            # Sum soft values across dimensions
            next_v = torch.zeros(s_.shape[0], device=self.device)
            for i in range(self.num_action_dims):
                min_q_i = torch.min(q1_next_list[i], q2_next_list[i])
                v_i = (next_probs_list[i] * (
                    min_q_i - alphas[i].detach() * next_log_probs_list[i]
                )).sum(-1)
                next_v = next_v + v_i

            target_q = r + self.gamma * next_v * t

        # ---- Q losses (sum of per-dimension gathered values) ----
        q1_list = self.q1(s)
        q2_list = self.q2(s)

        q1_val = sum(
            q1_list[i].gather(1, a[:, i:i+1]).squeeze(-1)
            for i in range(self.num_action_dims)
        )
        q2_val = sum(
            q2_list[i].gather(1, a[:, i:i+1]).squeeze(-1)
            for i in range(self.num_action_dims)
        )

        q1_loss = F.mse_loss(q1_val, target_q)
        q2_loss = F.mse_loss(q2_val, target_q)

        self.q1_optimizer.zero_grad(); q1_loss.backward()
        torch.nn.utils.clip_grad_value_(self.q1.parameters(), 100)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad(); q2_loss.backward()
        torch.nn.utils.clip_grad_value_(self.q2.parameters(), 100)
        self.q2_optimizer.step()

        # ---- Policy loss (sum over dimensions) ----
        probs_list, log_probs_list = self.policy.get_action_probs(s)
        q1_pi_list = self.q1(s)
        q2_pi_list = self.q2(s)

        policy_loss = torch.zeros(1, device=self.device)
        for i in range(self.num_action_dims):
            min_q_i = torch.min(q1_pi_list[i], q2_pi_list[i])
            policy_loss = policy_loss + (
                probs_list[i] * (alphas[i].detach() * log_probs_list[i] - min_q_i)
            ).sum(-1).mean()

        self.policy_optimizer.zero_grad(); policy_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.policy_optimizer.step()

        # ---- Alpha (per-dimension) ----
        if self.learnable_alpha:
            for i in range(self.num_action_dims):
                entropy_i = -(probs_list[i].detach() * log_probs_list[i].detach()).sum(-1).mean()
                alpha_loss_i = self.log_alphas[i] * (entropy_i - self.target_entropies[i])
                self.alpha_optimizers[i].zero_grad()
                alpha_loss_i.backward()
                self.alpha_optimizers[i].step()

        self.update_steps += 1
        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)

    def save_model(self, save_path):
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
        }, os.path.join(save_path, 'multi_sac_model.pth'))


# =============================================================================
# MaskSAC — single discrete
# =============================================================================

class MaskSAC(SAC):
    """Discrete SAC with action masking via learned mask, transition, and reward models."""
    def __init__(
            self,
            obs_n,
            act_n,
            motion,
            device,
            load_flag,
            policy_lr=3e-4,
            q_lr=3e-4,
            alpha_lr=3e-4,
            mask_lr=1e-3,
            transition_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            batch_size=256,
            replay_memory_size=2000,
            trans_freq=100,
            init_alpha=0.2,
            learnable_alpha=True,
    ):
        super().__init__(
            obs_n, act_n, device, policy_lr, q_lr, alpha_lr,
            gamma, tau, batch_size, replay_memory_size, trans_freq,
            init_alpha, learnable_alpha,
        )
        self.motion = motion

        self.mask_model = MaskModel(2, self.act_n, load_flag).to(device)
        self.transition_model = DeterministicTransitionModel(2, 2).to(device)
        self.reward_model = RewardModel(2, 2).to(device)

        self.mask_optimizer = torch.optim.Adam(self.mask_model.parameters(), lr=mask_lr)
        self.transition_optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=transition_lr)
        self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-4)

        self.fit_criterion = nn.MSELoss()
        self.reward_criterion = nn.MSELoss()

    def get_action(self, state, action=None, mask=None):
        mask, _ = self.mask_model.get_mask(state[:, :2])
        probs, log_probs = self.policy.get_action_probs(state, mask=mask)
        if action is None:
            action = torch.multinomial(probs, 1).squeeze(-1)
        return action, probs, mask

    def update_policy(self, s, a, r, s_, t):
        with torch.no_grad():
            next_mask, _ = self.mask_model.get_mask(s_[:, :2])
            next_probs, next_log_probs = self.policy.get_action_probs(s_, mask=next_mask)
            q1_next = self.q1_target(s_)
            q2_next = self.q2_target(s_)
            min_q_next = torch.min(q1_next, q2_next)
            next_v = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(-1)
            target_q = r + self.gamma * next_v * t

        a_idx = a.unsqueeze(-1) if a.dim() == 1 else a
        q1_vals = self.q1(s).gather(1, a_idx).squeeze(-1)
        q2_vals = self.q2(s).gather(1, a_idx).squeeze(-1)

        q1_loss = F.mse_loss(q1_vals, target_q)
        q2_loss = F.mse_loss(q2_vals, target_q)

        self.q1_optimizer.zero_grad(); q1_loss.backward()
        torch.nn.utils.clip_grad_value_(self.q1.parameters(), 100)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad(); q2_loss.backward()
        torch.nn.utils.clip_grad_value_(self.q2.parameters(), 100)
        self.q2_optimizer.step()

        curr_mask, _ = self.mask_model.get_mask(s[:, :2])
        probs, log_probs = self.policy.get_action_probs(s, mask=curr_mask)
        min_q_pi = torch.min(self.q1(s), self.q2(s))
        policy_loss = (probs * (self.alpha.detach() * log_probs - min_q_pi)).sum(-1).mean()

        self.policy_optimizer.zero_grad(); policy_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.policy_optimizer.step()

        if self.learnable_alpha:
            entropy = -(probs.detach() * log_probs.detach()).sum(-1).mean()
            alpha_loss = self.log_alpha * (entropy - self.target_entropy)
            self.alpha_optimizer.zero_grad(); alpha_loss.backward()
            self.alpha_optimizer.step()

        self.update_steps += 1
        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)

    def update_transition_reward(self, s, a, r, s_):
        s = s[:, :2]
        s_ = s_[:, :2]
        motion = np.zeros((s.shape[0], self.motion.shape[1]))
        for idx, act in enumerate(a):
            motion[idx] = self.motion[act]
        motion = torch.tensor(motion, dtype=torch.float32, device=self.device)

        pred_s_ = self.transition_model(torch.cat([s, motion], dim=1))
        diff = pred_s_ - s_.detach()
        transition_loss = torch.mean(0.5 * diff.pow(2))

        self.transition_optimizer.zero_grad(); transition_loss.backward()
        self.transition_optimizer.step()

        pred_reward = self.reward_model(torch.cat([s, motion], dim=1))
        reward_loss = self.reward_criterion(pred_reward, r.detach())

        self.reward_optimizer.zero_grad(); reward_loss.backward()
        self.reward_optimizer.step()

        return transition_loss, reward_loss

    def update_mask(self, s):
        simple_s = s[:, :2]
        all_motion = torch.tensor(self.motion, dtype=torch.float32, device=self.device)
        labels = []
        for state in simple_s:
            state_batch = state.unsqueeze(0).repeat(self.act_n, 1)
            with torch.no_grad():
                pred_s_ = self.transition_model(torch.cat([state_batch, all_motion], dim=1))
                pred_r = self.reward_model(torch.cat([state_batch, all_motion], dim=1)) / 1000
                dbscan = DBSCAN(eps=0.05, min_samples=1)
                clusters = dbscan.fit_predict(
                    torch.cat([pred_s_, pred_r], dim=1).cpu().numpy()
                )
            label, tmp = [], []
            for c in clusters:
                tag = 0 if c in tmp else 1
                tmp.append(c)
                label.append(tag)
            labels.append(label)
        labels = torch.tensor(np.array(labels), dtype=torch.float32).to(self.device)

        mask, probs = self.mask_model.get_mask(simple_s)
        prob_labels = 1 - torch.abs(mask - labels).to(self.device)
        mask_loss = (probs.exp() - prob_labels).pow(2).mean()

        self.mask_optimizer.zero_grad(); mask_loss.backward()
        self.mask_optimizer.step()

        return mask_loss

    def save_model(self, save_path):
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'mask': self.mask_model.state_dict(),
        }, os.path.join(save_path, 'mask_sac_model.pth'))

    def load_model(self, load_path):
        ckpt = torch.load(os.path.join(load_path, 'mask_sac_model.pth'), map_location='cpu')
        if 'mask' in ckpt:
            self.mask_model.load_state_dict(ckpt['mask'])


# =============================================================================
# MaskSAC — multi-discrete
# =============================================================================

class MultiDiscreteMaskSAC(MultiDiscreteSAC):
    """Multi-discrete SAC with per-dimension action masking.

    Each action dimension has its own MaskModel, transition model, and
    reward model. DBSCAN clustering is performed independently per dimension
    to determine which actions within that dimension are redundant.

    Args:
        obs_n:      observation dimension
        act_dims:   list of ints, e.g. [3, 4, 5]
        motions:    list of np.arrays, motions[i] has shape (act_dims[i], motion_dim)
                    mapping each action index in dimension i to a motion vector
        device:     torch device
        load_flag:  passed to MaskModel
    """
    def __init__(
            self,
            obs_n,
            act_dims,
            motions,
            device,
            load_flag,
            policy_lr=3e-4,
            q_lr=3e-4,
            alpha_lr=3e-4,
            mask_lr=1e-3,
            transition_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            batch_size=256,
            replay_memory_size=2000,
            trans_freq=100,
            init_alpha=0.2,
            learnable_alpha=True,
    ):
        super().__init__(
            obs_n, act_dims, device, policy_lr, q_lr, alpha_lr,
            gamma, tau, batch_size, replay_memory_size, trans_freq,
            init_alpha, learnable_alpha,
        )
        self.motions = motions  # list of (act_dims[i], motion_dim) arrays

        # Per-dimension auxiliary models
        self.mask_models = nn.ModuleList([
            MaskModel(2, d, load_flag) for d in self.act_dims
        ]).to(device)

        self.transition_models = nn.ModuleList([
            DeterministicTransitionModel(2, 2) for _ in self.act_dims
        ]).to(device)

        self.reward_models = nn.ModuleList([
            RewardModel(2, 2) for _ in self.act_dims
        ]).to(device)

        self.mask_optimizers = [
            torch.optim.Adam(self.mask_models[i].parameters(), lr=mask_lr)
            for i in range(self.num_action_dims)
        ]
        self.transition_optimizers = [
            torch.optim.Adam(self.transition_models[i].parameters(), lr=transition_lr)
            for i in range(self.num_action_dims)
        ]
        self.reward_optimizers = [
            torch.optim.Adam(self.reward_models[i].parameters(), lr=1e-4)
            for i in range(self.num_action_dims)
        ]

        self.fit_criterion = nn.MSELoss()
        self.reward_criterion = nn.MSELoss()

    def _get_masks(self, state_2d):
        """Get masks for all action dimensions. Returns list of mask tensors."""
        masks = []
        for i in range(self.num_action_dims):
            m, _ = self.mask_models[i].get_mask(state_2d)
            masks.append(m)
        return masks

    def get_action(self, state, action=None, masks=None):
        masks = self._get_masks(state[:, :2])
        probs_list, _ = self.policy.get_action_probs(state, masks=masks)
        if action is None:
            action = torch.stack(
                [torch.multinomial(p, 1).squeeze(-1) for p in probs_list], dim=-1
            )
        return action, probs_list, masks

    def update_policy(self, s, a, r, s_, t):
        """a: (batch, num_dims) LongTensor"""
        alphas = self.alphas()

        # ---- Q targets with masks ----
        with torch.no_grad():
            next_masks = self._get_masks(s_[:, :2])
            next_probs_list, next_log_probs_list = self.policy.get_action_probs(s_, masks=next_masks)
            q1_next_list = self.q1_target(s_)
            q2_next_list = self.q2_target(s_)

            next_v = torch.zeros(s_.shape[0], device=self.device)
            for i in range(self.num_action_dims):
                min_q_i = torch.min(q1_next_list[i], q2_next_list[i])
                v_i = (next_probs_list[i] * (
                    min_q_i - alphas[i].detach() * next_log_probs_list[i]
                )).sum(-1)
                next_v = next_v + v_i

            target_q = r + self.gamma * next_v * t

        q1_list = self.q1(s)
        q2_list = self.q2(s)
        q1_val = sum(
            q1_list[i].gather(1, a[:, i:i+1]).squeeze(-1)
            for i in range(self.num_action_dims)
        )
        q2_val = sum(
            q2_list[i].gather(1, a[:, i:i+1]).squeeze(-1)
            for i in range(self.num_action_dims)
        )

        q1_loss = F.mse_loss(q1_val, target_q)
        q2_loss = F.mse_loss(q2_val, target_q)

        self.q1_optimizer.zero_grad(); q1_loss.backward()
        torch.nn.utils.clip_grad_value_(self.q1.parameters(), 100)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad(); q2_loss.backward()
        torch.nn.utils.clip_grad_value_(self.q2.parameters(), 100)
        self.q2_optimizer.step()

        # ---- Policy with masks ----
        curr_masks = self._get_masks(s[:, :2])
        probs_list, log_probs_list = self.policy.get_action_probs(s, masks=curr_masks)
        q1_pi_list = self.q1(s)
        q2_pi_list = self.q2(s)

        policy_loss = torch.zeros(1, device=self.device)
        for i in range(self.num_action_dims):
            min_q_i = torch.min(q1_pi_list[i], q2_pi_list[i])
            policy_loss = policy_loss + (
                probs_list[i] * (alphas[i].detach() * log_probs_list[i] - min_q_i)
            ).sum(-1).mean()

        self.policy_optimizer.zero_grad(); policy_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.policy_optimizer.step()

        # ---- Alpha (per-dimension) ----
        if self.learnable_alpha:
            for i in range(self.num_action_dims):
                entropy_i = -(
                    probs_list[i].detach() * log_probs_list[i].detach()
                ).sum(-1).mean()
                alpha_loss_i = self.log_alphas[i] * (entropy_i - self.target_entropies[i])
                self.alpha_optimizers[i].zero_grad()
                alpha_loss_i.backward()
                self.alpha_optimizers[i].step()

        self.update_steps += 1
        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)

    def update_transition_reward(self, s, a, r, s_):
        """Train per-dimension transition and reward models.

        a: (batch, num_dims) — each column indexes into motions[i].
        """
        s_2d = s[:, :2]
        s__2d = s_[:, :2]
        total_trans_loss = torch.zeros(1, device=self.device)
        total_reward_loss = torch.zeros(1, device=self.device)

        for i in range(self.num_action_dims):
            motion_np = np.zeros((s_2d.shape[0], self.motions[i].shape[1]))
            for idx, act_idx in enumerate(a[:, i]):
                motion_np[idx] = self.motions[i][act_idx]
            motion_t = torch.tensor(motion_np, dtype=torch.float32, device=self.device)

            pred_s_ = self.transition_models[i](torch.cat([s_2d, motion_t], dim=1))
            diff = pred_s_ - s__2d.detach()
            t_loss = torch.mean(0.5 * diff.pow(2))

            self.transition_optimizers[i].zero_grad()
            t_loss.backward()
            self.transition_optimizers[i].step()

            # Reward model receives r / num_dims as target (decomposed reward)
            pred_r = self.reward_models[i](torch.cat([s_2d, motion_t], dim=1))
            r_target = r / self.num_action_dims
            r_loss = self.reward_criterion(pred_r, r_target.detach())

            self.reward_optimizers[i].zero_grad()
            r_loss.backward()
            self.reward_optimizers[i].step()

            total_trans_loss = total_trans_loss + t_loss.detach()
            total_reward_loss = total_reward_loss + r_loss.detach()

        return total_trans_loss, total_reward_loss

    def update_mask(self, s):
        simple_s = s[:, :2]
        total_mask_loss = torch.zeros(1, device=self.device)

        for i in range(self.num_action_dims):
            all_motion_i = torch.tensor(
                self.motions[i], dtype=torch.float32, device=self.device
            )
            labels = []
            for state in simple_s:
                state_batch = state.unsqueeze(0).repeat(self.act_dims[i], 1)
                with torch.no_grad():
                    pred_s_ = self.transition_models[i](
                        torch.cat([state_batch, all_motion_i], dim=1)
                    )
                    pred_r = self.reward_models[i](
                        torch.cat([state_batch, all_motion_i], dim=1)
                    ) / 1000
                    dbscan = DBSCAN(eps=0.05, min_samples=1)
                    clusters = dbscan.fit_predict(
                        torch.cat([pred_s_, pred_r], dim=1).cpu().numpy()
                    )
                label, tmp = [], []
                for c in clusters:
                    tag = 0 if c in tmp else 1
                    tmp.append(c)
                    label.append(tag)
                labels.append(label)

            labels = torch.tensor(np.array(labels), dtype=torch.float32).to(self.device)
            mask, probs = self.mask_models[i].get_mask(simple_s)
            prob_labels = 1 - torch.abs(mask - labels).to(self.device)
            m_loss = (probs.exp() - prob_labels).pow(2).mean()

            self.mask_optimizers[i].zero_grad()
            m_loss.backward()
            self.mask_optimizers[i].step()

            total_mask_loss = total_mask_loss + m_loss.detach()

        return total_mask_loss

    def save_model(self, save_path):
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'masks': self.mask_models.state_dict(),
            'transitions': self.transition_models.state_dict(),
            'rewards': self.reward_models.state_dict(),
        }, os.path.join(save_path, 'multi_mask_sac_model.pth'))

    def load_model(self, load_path):
        ckpt = torch.load(
            os.path.join(load_path, 'multi_mask_sac_model.pth'), map_location='cpu'
        )
        if 'masks' in ckpt:
            self.mask_models.load_state_dict(ckpt['masks'])
        if 'transitions' in ckpt:
            self.transition_models.load_state_dict(ckpt['transitions'])
        if 'rewards' in ckpt:
            self.reward_models.load_state_dict(ckpt['rewards'])
