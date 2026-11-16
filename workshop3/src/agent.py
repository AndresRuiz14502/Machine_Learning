"""Dueling Double DQN Agent and Experience Replay Buffer for Sequential Movie Recommendation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Dueling Deep Q-Network with LayerNorm trunk and separate Value/Advantage streams.

    Architecture:
        - Feature trunk: Linear(state_dim, hidden_dim) -> LayerNorm -> ReLU
                         -> Linear(hidden_dim, hidden_dim) -> LayerNorm -> ReLU
        - Value stream:  Linear(hidden_dim, hidden_dim // 2) -> ReLU -> Linear(hidden_dim // 2, 1)
        - Advantage stream: Linear(hidden_dim, hidden_dim // 2) -> ReLU -> Linear(hidden_dim // 2, action_dim)
        - Aggregation:   Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))
    """

    def __init__(
        self,
        state_dim: int = 250,
        action_dim: int = 50,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        half_hidden = max(1, self.hidden_dim // 2)

        self.feature_trunk = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, half_hidden),
            nn.ReLU(),
            nn.Linear(half_hidden, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, half_hidden),
            nn.ReLU(),
            nn.Linear(half_hidden, self.action_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layers with Kaiming uniform weights and zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) for a single state (state_dim,) or batch (B, state_dim)."""
        features = self.feature_trunk(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class SlotDuelingQNetwork(nn.Module):
    """Slot-equivariant Dueling Q-Network: one shared scorer for every candidate slot.

    The 250-D state is split into per-slot features x_a = [similarity, FunkSVD score, novelty,
    unavailable flag, slot rank] (from state[0:200]) and the 10 session progress scalars
    (state[240:250]). The raw user factor and session vector (state[200:240]) are not fed to
    the network: they identify each training user, which lets a network memorize the few
    training sessions instead of learning a scoring rule. Personalization still enters through
    the FunkSVD score and the feedback-updated session similarity of every slot.

    A context MLP embeds the scalars, a shared MLP scores every slot from [x_a, context], and
    the dueling head combines a state value with the mean-centred advantages:

        Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))

    Sharing the scorer across slots lets one transition on slot a inform the Q-values of all
    slots with similar features, instead of training 50 independent output units.
    """

    def __init__(
        self,
        state_dim: int = 250,
        action_dim: int = 50,
        hidden_dim: int = 128,
        latent_dim: int = 20,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_slot_features = 5
        self.context_start = 4 * self.action_dim + 2 * int(latent_dim)
        self.context_dim = self.state_dim - self.context_start
        if self.context_dim <= 0:
            raise ValueError("state_dim must hold 4 per-slot blocks, 2 latent vectors and scalars")
        ctx_hidden = max(16, self.hidden_dim // 2)

        self.context_net = nn.Sequential(
            nn.Linear(self.context_dim, ctx_hidden),
            nn.LayerNorm(ctx_hidden),
            nn.ReLU(),
        )
        self.slot_net = nn.Sequential(
            nn.Linear(self.n_slot_features + ctx_hidden, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
        )
        self.advantage_head = nn.Linear(self.hidden_dim // 2, 1)
        self.value_head = nn.Sequential(
            nn.Linear(ctx_hidden + self.hidden_dim // 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        rank = torch.linspace(0.0, 1.0, self.action_dim)
        self.register_buffer("slot_rank", rank)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
        m = self.action_dim
        slot_feats = state[:, : 4 * m].reshape(state.shape[0], 4, m).transpose(1, 2)  # (B, M, 4)
        rank = self.slot_rank.expand(state.shape[0], m).unsqueeze(-1)  # (B, M, 1)
        x_slots = torch.cat([slot_feats, rank], dim=-1)  # (B, M, 5)

        ctx = self.context_net(state[:, self.context_start :])  # (B, C)
        ctx_rep = ctx.unsqueeze(1).expand(-1, m, -1)  # (B, M, C)
        h = self.slot_net(torch.cat([x_slots, ctx_rep], dim=-1))  # (B, M, H/2)

        advantage = self.advantage_head(h).squeeze(-1)  # (B, M)
        value = self.value_head(torch.cat([ctx, h.mean(dim=1)], dim=-1))  # (B, 1)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values.squeeze(0) if squeeze else q_values


def build_q_network(architecture: str, state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
    """Return the Q-network for ``architecture`` ('flat' MLP over the whole state or 'slot')."""
    if architecture == "flat":
        return DuelingQNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    if architecture == "slot":
        return SlotDuelingQNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown Q-network architecture '{architecture}' (expected 'flat' or 'slot')")


class ReplayBuffer:
    """Preallocated NumPy Experience Replay Buffer for fast batch sampling onto GPU/CPU."""

    def __init__(
        self,
        capacity: int = 50000,
        state_dim: int = 250,
        action_dim: int = 50,
        seed: int = 42,
    ) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.next_valid_masks = np.ones((self.capacity, self.action_dim), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    @staticmethod
    def extract_valid_mask_from_state(
        state: np.ndarray,
        action_dim: int = 50,
    ) -> np.ndarray:
        """Extract valid action mask (1.0 = unvisited/valid, 0.0 = visited) from state[150:200]."""
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.shape[-1] >= 150 + action_dim:
            visited = state_arr[..., 150 : 150 + action_dim]
            mask = 1.0 - visited
        else:
            mask = np.ones(action_dim, dtype=np.float32)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool | float,
        next_valid_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Store a single transition in the circular buffer."""
        idx = self.ptr
        self.states[idx] = np.asarray(state, dtype=np.float32)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        self.next_states[idx] = next_state_arr
        self.dones[idx] = 1.0 if bool(done) else 0.0

        if next_valid_mask is None:
            mask = self.extract_valid_mask_from_state(next_state_arr, self.action_dim)
        else:
            mask = np.asarray(next_valid_mask, dtype=np.float32)
        self.next_valid_masks[idx] = mask

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # Convenience aliases
    add = push
    store = push

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniformly sample a mini-batch of transitions onto the target torch device."""
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        dev = torch.device(device)

        states = torch.as_tensor(self.states[indices], dtype=torch.float32, device=dev)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.int64, device=dev)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=dev)
        next_states = torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=dev)
        dones = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=dev)
        next_valid_masks = torch.as_tensor(
            self.next_valid_masks[indices], dtype=torch.float32, device=dev
        )
        return states, actions, rewards, next_states, dones, next_valid_masks

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    """Dueling Double DQN Agent with action masking, Huber loss, and soft/periodic target updates."""

    name: str = "DuelingDoubleDQN"

    def __init__(
        self,
        state_dim: int = 250,
        action_dim: int = 50,
        hidden_dim: int = 256,
        lr: float = 5e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        weight_decay: float = 1e-5,
        batch_size: int = 128,
        buffer_capacity: int = 50000,
        target_update_freq: int = 20,
        tau: float = 0.05,
        grad_clip: float = 1.0,
        seed: int = 42,
        device: Optional[torch.device | str] = None,
        architecture: str = "flat",
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.architecture = str(architecture)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.buffer_capacity = int(buffer_capacity)
        self.target_update_freq = int(target_update_freq)
        self.tau = float(tau)
        self.grad_clip = float(grad_clip)
        self.seed = int(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.rng = np.random.default_rng(self.seed)

        self._build_networks()

        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_capacity,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seed=self.seed,
        )
        # Convenience aliases for buffer access
        self.buffer = self.replay_buffer
        self.memory = self.replay_buffer

        self.update_steps = 0
        self.episodes_done = 0

    def _build_networks(self) -> None:
        """(Re)build policy/target networks and the optimizer for the current architecture."""
        self.policy_net = build_q_network(
            self.architecture, self.state_dim, self.action_dim, self.hidden_dim
        ).to(self.device)
        self.target_net = build_q_network(
            self.architecture, self.state_dim, self.action_dim, self.hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def reset(self) -> None:
        """No-op reset hook for compatibility with BasePolicy evaluation interfaces."""
        pass

    def _resolve_valid_mask(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        """Extract a guaranteed non-empty valid action mask (1.0 = valid, 0.0 = invalid)."""
        if valid_mask is None and info is not None and "valid_mask" in info:
            valid_mask = info["valid_mask"]

        if valid_mask is None:
            mask = ReplayBuffer.extract_valid_mask_from_state(state, self.action_dim)
        else:
            mask = np.asarray(valid_mask, dtype=np.float32).reshape(-1)

        if not np.any(mask > 0.5):
            mask = np.ones(self.action_dim, dtype=np.float32)
        return mask

    def get_q_values(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute Q-values (action_dim,) for state with invalid/visited actions masked to -1e9."""
        mask = self._resolve_valid_mask(state, valid_mask=valid_mask)
        state_arr = np.asarray(state, dtype=np.float32)
        with torch.no_grad():
            state_t = torch.as_tensor(state_arr, dtype=torch.float32, device=self.device)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            q_vals = self.policy_net(state_t).squeeze(0).detach().cpu().numpy()

        masked_q = np.where(mask > 0.5, q_vals, -1e9).astype(np.float32)
        return masked_q

    def select_action(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray | dict[str, Any]] = None,
        eval_mode: bool = False,
        info: Optional[dict[str, Any]] = None,
    ) -> int:
        """Select an action using epsilon-greedy exploration over valid actions.

        Also supports calling `agent.select_action(state, info)` (where second argument is an `info`
        dict) for drop-in compatibility with `evaluate_policy(env, policy)` in `baselines.py`.
        """
        if isinstance(valid_mask, dict):
            info = valid_mask
            valid_mask = info.get("valid_mask", None)
            eval_mode = True

        mask = self._resolve_valid_mask(state, valid_mask=valid_mask, info=info)
        valid_indices = np.flatnonzero(mask > 0.5)

        if (not eval_mode) and (self.rng.random() < self.epsilon):
            return int(self.rng.choice(valid_indices))

        q_values = self.get_q_values(state, valid_mask=mask)
        return int(np.argmax(q_values))

    def rank_actions(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        """Return candidate action indices (50,) sorted by descending masked Q-values."""
        mask = self._resolve_valid_mask(state, valid_mask=valid_mask, info=info)
        q_values = self.get_q_values(state, valid_mask=mask)
        return np.argsort(-q_values).astype(np.int64)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool | float,
        next_valid_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Store a transition `(s, a, r, s', done, next_valid_mask)` in the replay buffer."""
        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            next_valid_mask=next_valid_mask,
        )

    def update_target_network(self, hard: bool = False) -> None:
        """Update target network parameters via Polyak averaging (`tau`) or hard copy."""
        with torch.no_grad():
            if hard or self.tau >= 1.0 or self.tau <= 0.0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            else:
                for target_param, policy_param in zip(
                    self.target_net.parameters(), self.policy_net.parameters()
                ):
                    target_param.data.mul_(1.0 - self.tau).add_(policy_param.data, alpha=self.tau)

    def update(self) -> Optional[dict[str, float]]:
        """Sample a mini-batch and perform a Double DQN update with Huber loss."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_valid_masks = (
            self.replay_buffer.sample(self.batch_size, device=self.device)
        )

        # Current Q-values Q_online(s, a)
        q_all = self.policy_net(states)
        current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: online net selects best valid action, target net evaluates it
        with torch.no_grad():
            next_q_online = self.policy_net(next_states)
            valid_bool = next_valid_masks > 0.5
            any_valid = valid_bool.any(dim=1, keepdim=True)
            safe_valid_bool = torch.where(any_valid, valid_bool, torch.ones_like(valid_bool))
            masked_next_q_online = next_q_online.masked_fill(~safe_valid_bool, -1e9)

            next_actions = masked_next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + (1.0 - dones) * self.gamma * next_q_target

        loss = F.smooth_l1_loss(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm_tensor = nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.grad_clip
        )
        self.optimizer.step()

        self.update_steps += 1
        if self.tau > 0.0:
            self.update_target_network(hard=False)
        elif self.target_update_freq > 0 and (self.update_steps % self.target_update_freq == 0):
            self.update_target_network(hard=True)

        grad_norm_val = (
            float(grad_norm_tensor.item())
            if isinstance(grad_norm_tensor, torch.Tensor)
            else float(grad_norm_tensor)
        )
        # Valid-masked mean Q over the batch for diagnostic tracking
        return {
            "loss": float(loss.item()),
            "mean_q": float(current_q.detach().mean().item()),
            "grad_norm": grad_norm_val,
        }

    def decay_epsilon(self) -> float:
        """Decay exploration rate epsilon after an episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_done += 1
        if self.tau <= 0.0 and self.target_update_freq > 0:
            if self.episodes_done % self.target_update_freq == 0:
                self.update_target_network(hard=True)
        return self.epsilon

    def get_config(self) -> dict[str, Any]:
        """Return serializable agent hyperparameter configuration."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "architecture": self.architecture,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "buffer_capacity": self.buffer_capacity,
            "target_update_freq": self.target_update_freq,
            "tau": self.tau,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
        }

    def save(self, filepath: str | Path) -> None:
        """Save policy/target networks, optimizer state, and configuration to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_steps": self.update_steps,
            "episodes_done": self.episodes_done,
            "config": self.get_config(),
        }
        torch.save(checkpoint, path)

    def load(self, filepath: str | Path) -> dict[str, Any]:
        """Load policy/target networks, optimizer state, and configuration from disk."""
        path = Path(filepath)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        saved = checkpoint.get("config", {})
        saved_arch = str(saved.get("architecture", "flat"))
        saved_hidden = int(saved.get("hidden_dim", self.hidden_dim))
        if saved_arch != self.architecture or saved_hidden != self.hidden_dim:
            # Rebuild the networks so any checkpoint loads into a matching architecture
            self.architecture, self.hidden_dim = saved_arch, saved_hidden
            self._build_networks()
        for key in ("gamma", "lr", "weight_decay", "epsilon_decay"):
            if key in saved:
                setattr(self, key, float(saved[key]))
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        if "target_state_dict" in checkpoint:
            self.target_net.load_state_dict(checkpoint["target_state_dict"])
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epsilon" in checkpoint:
            self.epsilon = float(checkpoint["epsilon"])
        if "update_steps" in checkpoint:
            self.update_steps = int(checkpoint["update_steps"])
        if "episodes_done" in checkpoint:
            self.episodes_done = int(checkpoint["episodes_done"])
        self.target_net.eval()
        return checkpoint.get("config", self.get_config())
