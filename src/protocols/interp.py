"""Interpretability Protocol: agents propose and test circuit hypotheses via ablation."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.protocols.base import Protocol
from src.sandbox import Sandbox

# A small 2-layer attention model trained on copy-from-3-back patterns
MODEL_CODE = '''
"""A small 2-layer transformer trained on byte-level copy/shift patterns.

The training data has a "copy from position i-3" pattern with ~50% probability.
The model learns induction-head-like circuits to exploit this pattern.
Weights are cached after deterministic training (seed 1337, 2000 steps).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

VOCAB_SIZE = 64
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
SEQ_LEN = 32
COPY_OFFSET = 3
COPY_PROB = 0.5

class SmallTransformer(nn.Module):
    """2-layer transformer with 4 attention heads per layer.
    Trained on byte-level sequences where ~50% of tokens at position i
    are copied from position i-3. The model learns induction-head circuits
    to detect and predict these copy patterns."""

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, seq_len=SEQ_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "attn_qkv": nn.Linear(d_model, 3 * d_model),
                "attn_out": nn.Linear(d_model, d_model),
                "ln1": nn.LayerNorm(d_model),
                "ff1": nn.Linear(d_model, 4 * d_model),
                "ff2": nn.Linear(4 * d_model, d_model),
                "ln2": nn.LayerNorm(d_model),
            }))

        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, return_activations=False):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(positions)

        activations = {}
        for i, layer in enumerate(self.layers):
            # Self-attention
            residual = h
            h_norm = layer["ln1"](h)
            qkv = layer["attn_qkv"](h_norm)
            q, k, v = qkv.chunk(3, dim=-1)

            head_dim = self.d_model // self.n_heads
            q = q.view(B, T, self.n_heads, head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_heads, head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_heads, head_dim).transpose(1, 2)

            attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
            attn_probs = F.softmax(attn_weights, dim=-1)

            if return_activations:
                activations[f"layer_{i}_attn_probs"] = attn_probs.detach()
                activations[f"layer_{i}_q"] = q.detach()
                activations[f"layer_{i}_k"] = k.detach()
                activations[f"layer_{i}_v"] = v.detach()

            attn_out = (attn_probs @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
            h = residual + layer["attn_out"](attn_out)

            # FFN
            residual = h
            h_norm = layer["ln2"](h)
            h = residual + layer["ff2"](F.gelu(layer["ff1"](h_norm)))

            if return_activations:
                activations[f"layer_{i}_output"] = h.detach()
                activations[f"layer_{i}_mlp_out"] = (h - residual).detach()

        logits = self.unembed(self.ln_final(h))
        if return_activations:
            return logits, activations
        return logits


def _generate_copy_data(n_samples, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                        copy_offset=COPY_OFFSET, copy_prob=COPY_PROB):
    """Generate sequences where token at position i has copy_prob chance
    of being copied from position i - copy_offset."""
    x = torch.randint(0, vocab_size, (n_samples, seq_len))
    is_copy = torch.zeros(n_samples, seq_len, dtype=torch.bool)
    for pos in range(copy_offset, seq_len):
        mask = torch.rand(n_samples) < copy_prob
        x[mask, pos] = x[mask, pos - copy_offset]
        is_copy[mask, pos] = True
    return x, is_copy


def _get_cache_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_trained_weights.pt")


def _train_model():
    """Train the model for 2000 steps on copy-pattern data. Deterministic (seed 1337).
    Caches weights to disk so training only happens once."""
    cache_path = _get_cache_path()
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    torch.manual_seed(1337)
    import random
    random.seed(1337)

    device = torch.device("cpu")  # train on CPU for reproducibility
    model = SmallTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    n_steps = 2000
    batch_size = 64

    for step in range(n_steps):
        torch.manual_seed(1337 + step)
        x, _ = _generate_copy_data(batch_size)
        x = x.to(device)

        logits = model(x)
        # next-token prediction: predict x[:, 1:] from logits[:, :-1]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB_SIZE), x[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            first_loss = loss.item()
        if step == n_steps - 1:
            final_loss = loss.item()

    print(f"Training complete: loss {first_loss:.3f} -> {final_loss:.3f}")
    state = model.state_dict()
    torch.save(state, cache_path)
    return state


# Train at import time, cache weights
_TRAINED_WEIGHTS = _train_model()


def get_model():
    """Return the pre-trained model with learned copy-pattern circuits."""
    model = SmallTransformer()
    model.load_state_dict(_TRAINED_WEIGHTS)
    return model


def get_test_inputs(n_samples=64):
    """Generate test sequences with known copy-from-3-back patterns.
    Returns (inputs, is_copy_mask) where is_copy_mask[b, t] is True
    when position t is a copy of position t-3."""
    torch.manual_seed(42)
    x, is_copy = _generate_copy_data(n_samples)
    return x, is_copy


def get_copy_positions(x):
    """Identify positions where token at i equals token at i-3 (copy positions).
    Returns a boolean mask of shape (batch, seq_len)."""
    mask = torch.zeros_like(x, dtype=torch.bool)
    for pos in range(COPY_OFFSET, x.shape[1]):
        mask[:, pos] = (x[:, pos] == x[:, pos - COPY_OFFSET])
    return mask
'''

EVALUATE_CODE = '''
"""Evaluate interpretability experiment quality.

Scoring rewards experiments that:
1. Find components specifically affecting copy-pattern positions
2. Have bounded fitness (no explosive p-value hacking)
3. Report structured results (component, hypothesis, specificity)
"""
import importlib.util
import json
import sys
import torch
import math

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    model_mod = load_module("model", "model.py")
    experiment = load_module("experiment", "solution.py")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_mod.get_model().to(device).eval()
    test_inputs, is_copy_mask = model_mod.get_test_inputs()
    test_inputs = test_inputs.to(device)
    is_copy_mask = is_copy_mask.to(device)

    # Validate required interface
    if not hasattr(experiment, "run_experiment"):
        print(json.dumps({"fitness": 0.0, "error": "solution.py must define run_experiment(model, inputs, is_copy_mask)"}))
        sys.exit(0)

    # Call with both inputs and copy mask
    import inspect
    sig = inspect.signature(experiment.run_experiment)
    n_params = len(sig.parameters)
    if n_params >= 3:
        results = experiment.run_experiment(model, test_inputs, is_copy_mask)
    else:
        results = experiment.run_experiment(model, test_inputs)

    # --- Extract structured fields ---
    effect_size = abs(float(results.get("effect_size", 0.0)))
    p_value = float(results.get("p_value", 1.0))
    hypothesis = str(results.get("hypothesis", getattr(experiment, "hypothesis", "unknown")))
    description = str(results.get("description", ""))
    component = str(results.get("component", "unspecified"))

    # Specificity: effect on copy positions vs non-copy positions
    # Higher specificity = the ablation specifically affects copy circuits
    # Both effects should be positive (loss increase) for a meaningful ablation.
    # If non_copy_effect is negative (model improves), that's even more specific.
    copy_effect = float(results.get("copy_effect", 0.0))
    non_copy_effect = float(results.get("non_copy_effect", 0.0))
    if abs(non_copy_effect) > 1e-6:
        # If copy_effect > 0 and non_copy_effect <= 0, component only hurts copy positions
        if copy_effect > 1e-6 and non_copy_effect <= 1e-6:
            specificity = min(abs(copy_effect) / max(abs(non_copy_effect), 1e-6), 10.0)
        else:
            specificity = abs(copy_effect) / abs(non_copy_effect)
    elif abs(copy_effect) > 1e-6:
        specificity = 10.0  # max specificity if only copy positions affected
    else:
        specificity = 0.0

    is_specific = bool(results.get("is_specific_to_copy", specificity > 1.5))

    # --- Bounded fitness ---
    # fitness = specificity * min(effect_size, 10) * min(-log10(p_value), 20)
    if p_value <= 0:
        p_value = 1e-20
    p_value = max(p_value, 1e-20)
    significance = min(-math.log10(p_value), 20.0)
    capped_effect = min(effect_size, 10.0)

    # Specificity bonus: ranges from 0.1 (no specificity) to 5.0 (very specific)
    spec_factor = max(min(specificity, 5.0), 0.1)

    fitness = spec_factor * capped_effect * significance

    # Small bonus for good experimental practice
    has_control = bool(results.get("has_control", False))
    has_multiple_seeds = bool(results.get("has_multiple_seeds", False))
    bonus = 1.0
    if has_control:
        bonus += 0.1
    if has_multiple_seeds:
        bonus += 0.1
    fitness *= bonus

    print(json.dumps({
        "fitness": round(fitness, 6),
        "effect_size": round(effect_size, 6),
        "capped_effect": round(capped_effect, 6),
        "p_value": p_value,
        "significance": round(significance, 4),
        "specificity": round(specificity, 4),
        "is_specific_to_copy": is_specific,
        "component": component,
        "hypothesis": hypothesis,
        "description": description,
        "copy_effect": round(copy_effect, 6),
        "non_copy_effect": round(non_copy_effect, 6),
        "has_control": has_control,
        "has_multiple_seeds": has_multiple_seeds,
    }))

except Exception as e:
    import traceback
    print(json.dumps({"fitness": 0.0, "error": str(e), "traceback": traceback.format_exc()}))
'''


class InterpProtocol(Protocol):
    """Mechanistic interpretability probe tournament.

    Agents propose circuit hypotheses and write ablation experiments on a
    pre-trained transformer that learned copy-from-3-back patterns.
    Fitness = specificity * min(effect_size, 10) * min(-log10(p_value), 20).
    """

    def __init__(self, **kwargs):
        pass

    @property
    def name(self) -> str:
        return "interp"

    @property
    def fitness_key(self) -> str:
        return "fitness"

    @property
    def fitness_direction(self) -> str:
        return "max"

    def setup_workspace(self, sandbox: Sandbox, round_num: int, prior_best: Optional[str]) -> None:
        sandbox.write_file("model.py", MODEL_CODE)
        sandbox.write_file("_evaluate.py", EVALUATE_CODE)
        if prior_best:
            sandbox.write_file("prior_best.py", prior_best)

    def get_system_prompt(self) -> str:
        return """You are a mechanistic interpretability researcher competing in a tournament.

You are given a small 2-layer transformer (model.py) with 4 attention heads per layer.
The model was PRE-TRAINED on byte-level sequences with a "copy from position i-3" pattern:
~50% of tokens at each position are copies of the token 3 positions earlier.

WHAT THE MODEL LEARNED:
The model has learned induction-head-like circuits to exploit the copy-from-3-back pattern.
This means there are likely:
- Attention heads that look back exactly 3 positions (copy heads)
- Heads in earlier layers that create "previous token" signals (predecessor heads)
- MLP layers that may encode positional or token-identity features
- Composition between L0 heads and L1 heads forming induction circuits

YOUR GOAL: Find which specific components (layer X, head Y, or MLP Z) implement the
copy circuit. The key metric is SPECIFICITY -- does ablating the component hurt performance
on copy positions MORE than non-copy positions?

YOUR SOLUTION (solution.py) must define:
- run_experiment(model, inputs, is_copy_mask) -> dict with keys:
  - effect_size: float -- magnitude of the ablation/patching effect
  - p_value: float -- statistical significance
  - hypothesis: str -- what you think the component does
  - description: str -- what the experiment does
  - component: str -- which component was ablated (e.g. "layer_1_head_2", "layer_0_mlp")
  - copy_effect: float -- loss increase on copy positions when component is ablated
  - non_copy_effect: float -- loss increase on non-copy positions when component is ablated
  - is_specific_to_copy: bool -- whether the effect is specific to copy positions
  - has_control: bool -- whether a control condition was included
  - has_multiple_seeds: bool -- whether multiple random seeds were tested

SCORING: fitness = specificity * min(effect_size, 10) * min(-log10(p_value), 20)
where specificity = copy_effect / non_copy_effect (higher = more specific to copy circuit).

CRITICAL INSIGHTS:
- Specificity matters MORE than raw effect size. Zeroing any head hurts loss, but only
  copy-circuit heads hurt copy positions disproportionately.
- Effect size is capped at 10.0 and -log10(p_value) is capped at 20.0, so you cannot
  win by gaming p-values. Focus on finding the RIGHT component.
- Use model.get_test_inputs() which returns (inputs, is_copy_mask).
- Use model.get_copy_positions(x) to identify copy positions in any batch.
- The model has 2 layers x 4 heads. Try ablating each head individually and measuring
  the differential effect on copy vs non-copy positions.

STRATEGY:
1. First, visualize attention patterns -- which heads attend to position i-3?
2. Ablate individual heads (zero their output) and measure loss on copy vs non-copy positions
3. The head with the highest specificity ratio is likely the induction/copy head
4. Test composition: does ablating an L0 head hurt an L1 copy head's performance?

TOOLS: read_file, write_file, edit_file, bash, submit
Do NOT modify model.py or _evaluate.py.
"""

    def get_initial_message(
        self, round_num: int, prior_best: Optional[str], prior_fitness: Optional[float]
    ) -> str:
        msg = f"Round {round_num} of the interpretability tournament.\n\n"
        msg += "Read model.py to understand the transformer architecture.\n"
        msg += "The model is PRE-TRAINED on copy-from-3-back patterns. There are real circuits to find.\n\n"

        if prior_best and prior_fitness:
            msg += f"""The current best experiment achieved fitness={prior_fitness:.4f}.
It is saved as prior_best.py. You must design a MORE specific experiment.
Read what was tested before, then explore DIFFERENT components or deeper interactions.

Focus on:
- Which specific head(s) implement the copy circuit? (measure specificity)
- Composition between L0 and L1 heads (does an L0 head feed into an L1 copy head?)
- Path patching through the full induction circuit
- MLP contributions to copy vs non-copy positions
"""
        else:
            msg += """First round -- no prior experiments.
Start by reading model.py. Then:
1. Load the model and examine attention patterns on test data
2. Identify which heads attend to position i-3 (these are candidate copy heads)
3. Ablate individual heads and measure loss increase on copy vs non-copy positions
4. Report the most specific component you find

Write your experiment in solution.py. The evaluation will call:
  run_experiment(model, inputs, is_copy_mask) -> dict
"""
        return msg

    def evaluate(self, sandbox: Sandbox) -> Dict[str, Any]:
        result = sandbox.run_command("python3 _evaluate.py", timeout=180)
        stdout = result["stdout"].strip()
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {"fitness": 0.0, "error": f"Parse failed: {stdout[:500]}"}
