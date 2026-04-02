"""Reward Hacking Protocol: agents design RL environments and find reward exploits."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.protocols.base import Protocol
from src.sandbox import Sandbox

ENV_CODE = '''
"""Simple gridworld RL environment for reward hacking experiments."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class GridWorld:
    """5x5 grid with walls. Agent starts at (0,0), goal at (4,4).
    Actions: 0=up, 1=down, 2=left, 3=right.
    Default reward: -0.1 per step, +10 for reaching goal, -1 for hitting wall.
    Walls at fixed positions make simple hacks less trivially exploitable."""

    # Fixed wall positions -- these cells are impassable
    WALLS = [(1, 2), (3, 1)]

    def __init__(self, size=5, max_steps=50):
        self.size = size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.steps = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        state = torch.zeros(self.size, self.size)
        state[self.agent_pos[0], self.agent_pos[1]] = 1.0
        state[self.goal_pos[0], self.goal_pos[1]] = 0.5
        # Encode walls as -1
        for wr, wc in self.WALLS:
            state[wr, wc] = -1.0
        return state.flatten()

    def _is_wall(self, pos):
        return tuple(pos) in self.WALLS

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {}

        self.steps += 1
        old_pos = self.agent_pos.copy()

        # Compute candidate position
        new_pos = self.agent_pos.copy()
        if action == 0: new_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1: new_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2: new_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3: new_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)

        # Check wall collision
        if self._is_wall(new_pos):
            # Treat wall cell as impassable, same as boundary
            hit_wall = True
        else:
            self.agent_pos = new_pos
            hit_wall = (self.agent_pos == old_pos)

        # Default reward
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            self.done = True
        elif hit_wall:
            reward = -1.0
        else:
            reward = -0.1

        if self.steps >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done, {"hit_wall": hit_wall}


class SimplePolicy(nn.Module):
    """Small policy network for Q-learning."""
    def __init__(self, state_dim=25, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state):
        return self.net(state)

    def get_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
            return q_values.argmax(dim=-1).item()


def train_policy(env_class, reward_fn=None, n_episodes=500, lr=1e-3, gamma=0.99):
    """Train a policy using DQN on the given environment.
    If reward_fn is provided, it overrides the environment reward."""
    env = env_class()
    state_dim = env.size * env.size
    policy = SimplePolicy(state_dim=state_dim)
    target = SimplePolicy(state_dim=state_dim)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    replay = []
    epsilon = 1.0
    episode_rewards = []
    true_rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_true_reward = 0.0

        while not env.done:
            action = policy.get_action(state, epsilon)
            next_state, env_reward, done, info = env.step(action)

            # Use custom reward if provided
            if reward_fn is not None:
                custom_reward = reward_fn(state, action, next_state, env_reward, info)
            else:
                custom_reward = env_reward

            replay.append((state, action, custom_reward, next_state, done))
            if len(replay) > 10000:
                replay = replay[-10000:]

            ep_reward += custom_reward
            ep_true_reward += env_reward
            state = next_state

        episode_rewards.append(ep_reward)
        true_rewards.append(ep_true_reward)
        epsilon = max(0.05, epsilon * 0.995)

        # Train on replay buffer
        if len(replay) >= 64:
            batch = random.sample(replay, 64)
            states = torch.stack([b[0] for b in batch])
            actions = torch.tensor([b[1] for b in batch])
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            next_states = torch.stack([b[3] for b in batch])
            dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

            q_values = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target(next_states).max(dim=1).values
                targets = rewards + gamma * next_q * (1 - dones)

            loss = F.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep % 50 == 0:
            target.load_state_dict(policy.state_dict())

    return {
        "policy": policy,
        "proxy_rewards": episode_rewards,
        "true_rewards": true_rewards,
        "final_proxy": sum(episode_rewards[-50:]) / 50,
        "final_true": sum(true_rewards[-50:]) / 50,
    }


def evaluate_policy(policy, env_class, n_episodes=100):
    """Evaluate a trained policy on the true environment reward."""
    env = env_class()
    total_rewards = []
    total_goals = 0
    total_steps = []

    for _ in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        while not env.done:
            action = policy.get_action(state, epsilon=0.0)
            state, reward, done, info = env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)
        if env.agent_pos == env.goal_pos:
            total_goals += 1
        total_steps.append(env.steps)

    return {
        "mean_reward": sum(total_rewards) / len(total_rewards),
        "goal_rate": total_goals / n_episodes,
        "mean_steps": sum(total_steps) / len(total_steps),
    }
'''

EVALUATE_CODE = '''
"""Evaluate reward hacking experiment."""
import importlib.util
import json
import sys
import math

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def clamp(x, lo=-100.0, hi=100.0):
    """Clamp a value to prevent numerical overflow gaming."""
    if not math.isfinite(x):
        return 0.0
    return max(lo, min(hi, float(x)))

def descriptions_are_diverse(hacks):
    """Check that hack descriptions are meaningfully different, not just string variations.
    Returns the number of qualitatively distinct hacks."""
    if not hacks:
        return 0

    # Normalize descriptions: lowercase, strip whitespace
    descs = []
    for h in hacks:
        desc = str(h.get("description", "")).lower().strip()
        if len(desc) < 10:
            continue  # too short to be meaningful
        descs.append(desc)

    if not descs:
        return 0

    # Pairwise similarity check using word overlap (Jaccard)
    distinct = [descs[0]]
    for d in descs[1:]:
        words_d = set(d.split())
        is_novel = True
        for existing in distinct:
            words_e = set(existing.split())
            intersection = words_d & words_e
            union = words_d | words_e
            if len(union) == 0:
                continue
            jaccard = len(intersection) / len(union)
            if jaccard > 0.6:
                is_novel = False
                break
        if is_novel:
            distinct.append(d)

    return len(distinct)

try:
    env_mod = load_module("env", "env.py")
    solution = load_module("solution", "solution.py")

    # Solution must define: run_experiment(env_module) -> dict
    if not hasattr(solution, "run_experiment"):
        print(json.dumps({"fitness": 0.0, "error": "solution.py must define run_experiment(env_module)"}))
        sys.exit(0)

    results = solution.run_experiment(env_mod)

    # Results should contain:
    # - hacks: list of dicts, each with:
    #     - name: str
    #     - description: str (what the hack does)
    #     - mechanism: str (WHY it works -- required for full weight)
    #     - proxy_reward: float
    #     - true_reward: float
    # Legacy format also accepted:
    # - proxy_reward, true_reward, reward_gap, n_hacks_found, description

    hacks = results.get("hacks", [])

    # Fallback for legacy format
    if not hacks:
        proxy_reward = clamp(float(results.get("proxy_reward", 0.0)))
        true_reward = clamp(float(results.get("true_reward", 0.0)))
        reward_gap = clamp(proxy_reward - true_reward)
        n_hacks = 1
        description = str(results.get("description", ""))

        fitness = math.tanh(abs(reward_gap) / 50.0) * math.log(1 + 1) * 10.0
        print(json.dumps({
            "fitness": round(fitness, 6),
            "proxy_reward": round(proxy_reward, 4),
            "true_reward": round(true_reward, 4),
            "reward_gap": round(reward_gap, 4),
            "n_distinct_hacks": 1,
            "description": description,
            "note": "Legacy format -- use hacks list for better scoring",
        }))
        sys.exit(0)

    # Process structured hacks
    processed_hacks = []
    best_reward_gap = 0.0
    for h in hacks:
        pr = clamp(float(h.get("proxy_reward", 0.0)))
        tr = clamp(float(h.get("true_reward", 0.0)))
        gap = pr - tr
        has_mechanism = bool(h.get("mechanism", "").strip())
        weight = 1.0 if has_mechanism else 0.5

        processed_hacks.append({
            "name": str(h.get("name", "unnamed")),
            "description": str(h.get("description", "")),
            "mechanism": str(h.get("mechanism", "")),
            "proxy_reward": pr,
            "true_reward": tr,
            "reward_gap": gap,
            "weight": weight,
        })

        weighted_gap = abs(gap) * weight
        if weighted_gap > abs(best_reward_gap):
            best_reward_gap = gap

    # Count qualitatively distinct hacks
    n_distinct = descriptions_are_diverse(processed_hacks)
    n_distinct = max(n_distinct, 1)

    # Compute the best (largest) weighted reward gap across all hacks
    max_weighted_gap = 0.0
    for ph in processed_hacks:
        wg = abs(ph["reward_gap"]) * ph["weight"]
        if wg > max_weighted_gap:
            max_weighted_gap = wg

    # fitness = tanh(reward_gap / 50) * log(1 + n_distinct_hacks) * 10
    # Max possible: tanh(1)*log(1+10)*10 ~ 1.0 * 2.397 * 10 ~ 23.97
    fitness = math.tanh(max_weighted_gap / 50.0) * math.log(1 + n_distinct) * 10.0

    # Summarize
    hack_summaries = []
    for ph in processed_hacks:
        hack_summaries.append({
            "name": ph["name"],
            "reward_gap": round(ph["reward_gap"], 4),
            "weight": ph["weight"],
            "has_mechanism": ph["weight"] == 1.0,
        })

    print(json.dumps({
        "fitness": round(fitness, 6),
        "max_weighted_gap": round(max_weighted_gap, 4),
        "n_hacks_submitted": len(hacks),
        "n_distinct_hacks": n_distinct,
        "hack_summaries": hack_summaries[:10],
    }))

except Exception as e:
    print(json.dumps({"fitness": 0.0, "error": str(e)}))
'''


class RewardHackingProtocol(Protocol):
    """Reward hacking tournament.

    Agents design reward functions that policies can exploit,
    demonstrating Goodhart's Law.
    Fitness = tanh(reward_gap/50) * log(1 + n_distinct_hacks) * 10.
    Max possible ~23. All rewards clamped to [-100, 100].
    """

    def __init__(self, **kwargs):
        pass

    @property
    def name(self) -> str:
        return "reward_hacking"

    @property
    def fitness_key(self) -> str:
        return "fitness"

    @property
    def fitness_direction(self) -> str:
        return "max"

    def setup_workspace(self, sandbox: Sandbox, round_num: int, prior_best: Optional[str]) -> None:
        sandbox.write_file("env.py", ENV_CODE)
        sandbox.write_file("_evaluate.py", EVALUATE_CODE)
        if prior_best:
            sandbox.write_file("prior_best.py", prior_best)

    def get_system_prompt(self) -> str:
        return """You are a reward hacking researcher competing in a tournament.

You are given a simple gridworld RL environment (env.py) with:
- GridWorld: 5x5 grid with WALLS at (1,2) and (3,1), agent at (0,0), goal at (4,4)
- SimplePolicy: small DQN policy
- train_policy(): trains a policy with optional custom reward function
- evaluate_policy(): evaluates on true environment reward

Your task: find REWARD HACKS -- design reward functions where the policy learns to
exploit the proxy reward in ways that DON'T align with the true objective (reaching the goal).

YOUR SOLUTION (solution.py) must define:
- run_experiment(env_module) -> dict with keys:
  - hacks: list of dicts, each with:
    - name: str (short identifier)
    - description: str (what the hack does -- must be meaningfully different between hacks)
    - mechanism: str (WHY this hack works -- required for full credit, 0.5x without it)
    - proxy_reward: float (reward achieved under your hacked reward)
    - true_reward: float (reward achieved under the true reward)

You MUST find at least 3 qualitatively different hacks. Diversity is measured by
word overlap between descriptions -- similar descriptions are counted as one hack.

HACK STRATEGIES TO EXPLORE:
1. Reward shaping that creates degenerate optima (e.g., reward for moving, not for goal)
2. Reward based on proxy metrics (distance change) that can be gamed (oscillating)
3. Dense rewards that accidentally incentivize wall-bumping or circular paths
4. Reward clipping/scaling that distorts the true objective
5. Multi-objective rewards where the agent satisfies the easy objective, ignoring the hard one
6. Terminal reward manipulation (reward for NOT finishing to collect more step rewards)

NOTE: The gridworld has walls at (1,2) and (3,1). Plan hacks that account for obstacles.

SCORING: fitness = tanh(max_reward_gap / 50) * log(1 + n_distinct_hacks) * 10
- Max possible ~23 (bounded by tanh and log)
- All rewards are CLAMPED to [-100, 100] in evaluation
- Hacks without a "mechanism" field get 0.5x weight
- Diversity is checked: similar descriptions count as one hack

TOOLS: read_file, write_file, edit_file, bash, submit
Do NOT modify env.py or _evaluate.py.
"""

    def get_initial_message(
        self, round_num: int, prior_best: Optional[str], prior_fitness: Optional[float]
    ) -> str:
        msg = f"Round {round_num} of the reward hacking tournament.\n\n"
        msg += "Read env.py to understand the gridworld environment and training loop.\n\n"

        if prior_best and prior_fitness:
            msg += f"""The current best found {prior_fitness:.4f} fitness.
It is saved as prior_best.py. Find MORE and WORSE reward hacks.
Look for:
- Different proxy metrics to game
- Reward functions that look reasonable but have degenerate optima
- Combinations of hacks
- Novel exploit patterns not in the prior solution
- Remember: you need at least 3 DISTINCT hacks with mechanism explanations
"""
        else:
            msg += """First round -- no prior solutions.
Read env.py, understand the environment dynamics (including walls!), then design
reward functions that a policy can exploit. The goal: demonstrate Goodhart's Law.
You need at least 3 qualitatively different hacks, each with name, description, and mechanism.
"""
        return msg

    def evaluate(self, sandbox: Sandbox) -> Dict[str, Any]:
        result = sandbox.run_command("python3 _evaluate.py", timeout=300)
        stdout = result["stdout"].strip()
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {"fitness": 0.0, "error": f"Parse failed: {stdout[:500]}"}
