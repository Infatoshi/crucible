"""Scaling Laws Protocol: agents design tiny model configs and train them, fitting scaling curves."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.protocols.base import Protocol
from src.sandbox import Sandbox

TRAINING_CODE = '''
"""Tiny model training harness for scaling law experiments."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class TinyGPT(nn.Module):
    """Minimal GPT for scaling experiments. Configurable width, depth, heads."""
    def __init__(self, vocab_size=256, d_model=64, n_heads=4, n_layers=2, seq_len=128):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn_qkv": nn.Linear(d_model, 3 * d_model),
                "attn_out": nn.Linear(d_model, d_model),
                "ln2": nn.LayerNorm(d_model),
                "ff1": nn.Linear(d_model, 4 * d_model),
                "ff2": nn.Linear(4 * d_model, d_model),
            }))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos_embed(torch.arange(T, device=x.device))
        for layer in self.layers:
            residual = h
            h_n = layer["ln1"](h)
            qkv = layer["attn_qkv"](h_n)
            q, k, v = qkv.chunk(3, dim=-1)
            hs = self.d_model // (3 * self.d_model // self.d_model)  # head_size
            n_h = self.d_model // max(hs, 1) if hs > 0 else 1
            # Simple scaled dot-product attention
            scale = math.sqrt(self.d_model)
            attn = (q @ k.transpose(-2, -1)) / scale
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            h = residual + layer["attn_out"](attn @ v)
            residual = h
            h_n = layer["ln2"](h)
            h = residual + layer["ff2"](F.gelu(layer["ff1"](h_n)))
        return self.head(self.ln_f(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def generate_data(n_samples=1024, seq_len=128, vocab_size=256, seed=42):
    """Generate byte-level text-like data with patterns.
    Use different seeds for train/val/test splits to prevent memorization."""
    torch.manual_seed(seed)
    # Simple pattern: shifted sequences with some noise
    data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
    # Add copy pattern: position i sometimes copies position i-3
    for i in range(3, seq_len + 1):
        mask = torch.rand(n_samples) > 0.5
        data[mask, i] = data[mask, i - 3]
    return data[:, :-1], data[:, 1:]  # input, target


def train_model(model, train_x, train_y, val_x, val_y, max_seconds=60, lr=3e-4, batch_size=64):
    """Train for max_seconds, return final val loss."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    n_train = train_x.shape[0]
    steps = 0
    t0 = time.time()

    while time.time() - t0 < max_seconds:
        model.train()
        idx = torch.randint(0, n_train, (batch_size,))
        x = train_x[idx].to(device)
        y = train_y[idx].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        steps += 1

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(0, val_x.shape[0], batch_size):
            x = val_x[i:i+batch_size].to(device)
            y = val_y[i:i+batch_size].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            val_losses.append(loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    return {
        "val_loss": avg_val_loss,
        "steps": steps,
        "params": model.count_params(),
        "elapsed": time.time() - t0,
    }
'''

EVALUATE_CODE = '''
"""Evaluate scaling law experiment."""
import importlib.util
import json
import sys
import math
import torch
import torch.nn.functional as F

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    harness = load_module("harness", "harness.py")
    solution = load_module("solution", "solution.py")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Solution must define: run_experiment() -> dict
    if not hasattr(solution, "run_experiment"):
        print(json.dumps({"fitness": 0.0, "error": "solution.py must define run_experiment()"}))
        sys.exit(0)

    results = solution.run_experiment(harness, device)

    # Results should contain:
    # - configs: list of {params, val_loss, ...} dicts
    # - best_val_loss: float
    # - prediction_error: float (how well the scaling curve fits)
    # - scaling_prediction: dict mapping param_count -> predicted_loss
    configs = results.get("configs", [])
    best_val_loss = results.get("best_val_loss", float("inf"))
    prediction_error = results.get("prediction_error", float("inf"))
    scaling_prediction = results.get("scaling_prediction", {})

    # --- Hidden test set evaluation ---
    # The agent only gets seeds 42 (train) and 123 (val).
    # We evaluate on seed 789 (test) which the agent never sees.
    test_x, test_y = harness.generate_data(n_samples=512, seq_len=128, vocab_size=256, seed=789)

    # Rebuild the best model config and measure test loss.
    # The agent must return best_config with model hyperparams.
    best_config = results.get("best_config", {})
    d_model = best_config.get("d_model", 64)
    n_heads = best_config.get("n_heads", 4)
    n_layers = best_config.get("n_layers", 2)
    seq_len = best_config.get("seq_len", 128)

    # If the agent returned a trained model state_dict, use it
    best_model = harness.TinyGPT(
        vocab_size=256, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, seq_len=seq_len
    ).to(device)

    if "best_state_dict" in results:
        best_model.load_state_dict(results["best_state_dict"])
    else:
        # Retrain briefly with the best config using train data (seed 42)
        train_x, train_y = harness.generate_data(n_samples=1024, seq_len=128, seed=42)
        val_x, val_y = harness.generate_data(n_samples=256, seq_len=128, seed=123)
        harness.train_model(best_model, train_x, train_y, val_x, val_y,
                            max_seconds=30, lr=best_config.get("lr", 3e-4),
                            batch_size=best_config.get("batch_size", 64))

    # Compute test loss
    best_model.eval()
    test_losses = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, test_x.shape[0], batch_size):
            x = test_x[i:i+batch_size].to(device)
            y = test_y[i:i+batch_size].to(device)
            logits = best_model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            test_losses.append(loss.item())
    test_loss = sum(test_losses) / len(test_losses) if test_losses else float("inf")

    # --- Anti-memorization check ---
    memorization_penalty = 1.0
    if best_val_loss > 0 and test_loss > 2.0 * best_val_loss:
        memorization_penalty = 0.5  # likely memorizing train data

    # --- Suspiciously low test loss check ---
    if test_loss < 0.01:
        # Almost certainly gaming the evaluation
        print(json.dumps({
            "fitness": 0.0,
            "test_loss": round(test_loss, 6),
            "best_val_loss": round(best_val_loss, 6),
            "error": "Suspiciously low test_loss (<0.01), likely gaming evaluation.",
        }))
        sys.exit(0)

    # --- Scaling prediction evaluation ---
    prediction_bonus = 1.0
    if scaling_prediction and len(scaling_prediction) >= 2:
        # Test predictions against actual measured losses from configs
        config_lookup = {c["params"]: c["val_loss"] for c in configs if "params" in c and "val_loss" in c}
        errors = []
        for param_str, predicted_loss in scaling_prediction.items():
            param_count = int(param_str)
            if param_count in config_lookup:
                actual_loss = config_lookup[param_count]
                if actual_loss > 0:
                    rel_error = abs(predicted_loss - actual_loss) / actual_loss
                    errors.append(rel_error)
        if errors:
            mean_error = sum(errors) / len(errors)
            mean_error = min(mean_error, 1.0)  # cap at 1.0
            prediction_bonus = 1.0 + 0.5 * (1.0 - mean_error)

    # --- Fitness calculation ---
    # fitness = (1 / test_loss) * prediction_bonus * memorization_penalty
    raw_fitness = (1.0 / max(test_loss, 0.01)) * prediction_bonus * memorization_penalty

    # Cap fitness at 100.0
    fitness = min(raw_fitness, 100.0)

    print(json.dumps({
        "fitness": round(fitness, 6),
        "test_loss": round(test_loss, 6),
        "best_val_loss": round(best_val_loss, 6),
        "prediction_bonus": round(prediction_bonus, 4),
        "memorization_penalty": memorization_penalty,
        "prediction_error": round(prediction_error, 6) if prediction_error != float("inf") else None,
        "n_configs_tested": len(configs),
        "n_scaling_predictions": len(scaling_prediction),
        "configs": configs[:10],  # truncate for logging
    }))

except Exception as e:
    print(json.dumps({"fitness": 0.0, "error": str(e)}))
'''


class ScalingLawsProtocol(Protocol):
    """Scaling law micro-experiments.

    Agents design model configurations, train them on a fixed budget,
    and try to find the best loss for the compute budget.
    Fitness = (1/test_loss) * prediction_bonus * memorization_penalty, capped at 100.
    """

    def __init__(self, train_seconds: int = 60, **kwargs):
        self.train_seconds = train_seconds

    @property
    def name(self) -> str:
        return "scaling"

    @property
    def fitness_key(self) -> str:
        return "fitness"

    @property
    def fitness_direction(self) -> str:
        return "max"

    def setup_workspace(self, sandbox: Sandbox, round_num: int, prior_best: Optional[str]) -> None:
        sandbox.write_file("harness.py", TRAINING_CODE)
        sandbox.write_file("_evaluate.py", EVALUATE_CODE)
        if prior_best:
            sandbox.write_file("prior_best.py", prior_best)

    def get_system_prompt(self) -> str:
        return f"""You are a scaling laws researcher competing in a tournament.

You are given a training harness (harness.py) with:
- TinyGPT: a configurable small GPT model
- generate_data(seed=...): creates data splits. Use seed=42 for train, seed=123 for val.
- train_model(): trains for a fixed time budget and returns val_loss

Your task: write solution.py that defines run_experiment(harness, device) -> dict

The function should:
1. Generate train data with seed=42 and val data with seed=123
2. Try different model configurations (width, depth, heads, learning rate, batch size)
3. Train each config for {self.train_seconds}s
4. Fit a scaling curve to the results (loss vs params)
5. Return the best configuration, scaling predictions, and the trained model state_dict

YOUR SOLUTION must return a dict with:
- configs: list of dicts with "params", "val_loss", and config details
- best_val_loss: float (lowest val_loss achieved)
- best_config: dict with "d_model", "n_heads", "n_layers", "seq_len", "lr", "batch_size"
- best_state_dict: the state_dict of your best trained model
- prediction_error: float (MSE of scaling curve fit, optional)
- scaling_prediction: dict mapping param_count (as int key) to predicted_loss (float)
  Must have at least 2 entries. These predictions are tested against actual measured losses.

IMPORTANT NOTES:
- You only have access to train (seed=42) and val (seed=123) data.
- Evaluation uses a HIDDEN test set (different seed) that you never see.
- If test_loss > 2x val_loss, a 0.5x memorization penalty is applied.
- If test_loss < 0.01, fitness is set to 0 (suspicious).
- Fitness is capped at 100.0.
- Your scaling_prediction accuracy gives up to 1.5x bonus.

SCORING: fitness = (1/test_loss) * prediction_bonus * memorization_penalty
  prediction_bonus = 1.0 + 0.5 * (1 - mean_relative_error) if predictions provided

STRATEGIES:
- Chinchilla-style: find optimal width/depth ratio for fixed compute
- Test width vs depth tradeoff (wider shallow vs narrow deep)
- Learning rate scaling: does optimal LR change with model size?
- Architecture modifications to TinyGPT (change the model class)
- Focus on GENERALIZATION, not just training loss

TOOLS: read_file, write_file, edit_file, bash, submit
Do NOT modify harness.py or _evaluate.py.
Total experiment budget: ~5 minutes.
"""

    def get_initial_message(
        self, round_num: int, prior_best: Optional[str], prior_fitness: Optional[float]
    ) -> str:
        msg = f"Round {round_num} of the scaling laws tournament.\n\n"
        msg += "Read harness.py to understand the training setup and model architecture.\n\n"

        if prior_best and prior_fitness:
            msg += f"""The current best achieved fitness={prior_fitness:.4f}.
It is saved as prior_best.py. Beat it by finding better model configs or training strategies.
Consider:
- Different width/depth ratios
- Modified attention or FFN architecture
- Better learning rate schedules
- Data curriculum strategies
- Better scaling curve predictions for the prediction bonus
"""
        else:
            msg += """First round -- no prior results.
Read harness.py, then systematically explore the config space.
Start with a few diverse configs (small/wide, large/narrow, etc.),
measure val_loss, then iterate on the most promising direction.
Remember to return scaling_prediction and best_state_dict for full credit.
"""
        return msg

    def evaluate(self, sandbox: Sandbox) -> Dict[str, Any]:
        result = sandbox.run_command("python3 _evaluate.py", timeout=600)
        stdout = result["stdout"].strip()
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {"fitness": 0.0, "error": f"Parse failed: {stdout[:500]}"}
