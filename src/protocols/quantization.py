"""Quantization Protocol: agents quantize Qwen3-4B while preserving quality.

Target: Qwen/Qwen3-4B (4.02B params, 8.04 GB fp16, dense text-generation).
Fitness = quality * log2(compression_ratio), where quality = exp(-delta_ppl_norm).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from src.protocols.base import Protocol
from src.sandbox import Sandbox

# Pre-computed reference model stats
MODEL_ID = "Qwen/Qwen3-4B"
REF_PARAMS = 4_022_468_096
REF_BYTES_FP16 = 8_044_936_192  # sum(p.numel() * 2) for all params


REFERENCE_CODE = '''
"""Reference: Qwen/Qwen3-4B (dense, 4.02B params, text-generation).

Do NOT modify this file.

Model specs:
- Architecture: Qwen3 dense transformer (GQA, SwiGLU, RoPE)
- Parameters: 4,022,468,096
- FP16 size: 8,044,936,192 bytes (8.04 GB)
- Hidden: 2560, Layers: 36, Heads: 32 (8 KV), Intermediate: 9728
- Vocab: 151,936 tokens, Max context: 40,960 tokens
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"
REF_BYTES = 8_044_936_192

def load_model(device="cuda"):
    """Load reference model in fp16."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map=device
    )
    model.eval()
    return model

def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)

def get_ref_bytes():
    return REF_BYTES

def get_model_config():
    """Architecture details for sensitivity analysis."""
    return {
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9728,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "total_params": 4_022_468_096,
        "fp16_bytes": 8_044_936_192,
    }

def get_layer_sizes():
    """Per-layer parameter counts and bytes (fp16) for mixed-precision planning."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    layers = {}
    for name, p in model.named_parameters():
        # Group by layer
        parts = name.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            group = f"layers.{parts[idx+1]}.{parts[idx+2]}"
        else:
            group = parts[0] if len(parts) > 1 else name
        if group not in layers:
            layers[group] = {"params": 0, "bytes": 0, "shapes": []}
        layers[group]["params"] += p.numel()
        layers[group]["bytes"] += p.numel() * p.element_size()
        layers[group]["shapes"].append((name, list(p.shape)))
    del model
    return layers
'''


# Evaluation texts: diverse English passages for perplexity measurement.
# Fixed and deterministic -- agents cannot see these in advance.
# ~4000 tokens total across 8 passages covering different domains.
EVAL_TEXTS = [
    # Science
    "The discovery of gravitational waves in 2015 confirmed a major prediction of Einstein's general theory of relativity. The Laser Interferometer Gravitational-Wave Observatory, known as LIGO, detected ripples in spacetime caused by the merger of two black holes approximately 1.3 billion light-years away. Each black hole was roughly 30 times the mass of our Sun, and their collision released more energy than all the stars in the observable universe combined, albeit in the form of gravitational radiation rather than light. The signal lasted less than a second but carried information about the final moments before the merger. This achievement required decades of technological development, including mirrors polished to atomic smoothness and laser beams split and recombined with extraordinary precision. The detection opened an entirely new window on the universe, complementing electromagnetic observations with a fundamentally different kind of information about cosmic events.",

    # History
    "The printing press, developed by Johannes Gutenberg around 1440, transformed European society in ways that took centuries to fully manifest. Before movable type, books were copied by hand in monastic scriptoria, making each volume expensive and rare. A single Bible might take a scribe two years to produce. Gutenberg's innovation allowed hundreds of identical copies to be printed in the time previously needed for one. Within fifty years of the first printed Bible, an estimated twenty million volumes had been produced across Europe. Literacy rates began climbing as books became affordable for merchants and artisans, not just clergy and nobility. The standardization of texts reduced the gradual corruption that crept into hand-copied manuscripts, while the ability to widely distribute identical editions enabled scholars in different cities to reference the same page and line of a work. The Protestant Reformation, the Scientific Revolution, and the Enlightenment all depended in part on this technology of mass communication.",

    # Mathematics
    "The prime numbers have fascinated mathematicians for over two thousand years, yet fundamental questions about their distribution remain open. Euclid proved around 300 BCE that there are infinitely many primes, using an elegant argument by contradiction. If you assume a finite list of all primes and multiply them together, then add one, the result is either prime itself or divisible by a prime not on your list. The prime number theorem, proved independently by Hadamard and de la Vallee Poussin in 1896, established that the number of primes less than n is approximately n divided by the natural logarithm of n. The Riemann hypothesis, proposed in 1859 and still unproved, makes a much more precise claim about the distribution of primes by connecting them to the zeros of a complex-valued function. A proof would have profound implications not only for number theory but also for cryptography, since the security of RSA encryption depends on the practical difficulty of factoring large numbers into their prime components.",

    # Biology
    "CRISPR-Cas9 gene editing technology, adapted from a bacterial immune system, has revolutionized molecular biology since its development as a genome editing tool in 2012. Bacteria use CRISPR sequences as a memory of past viral infections, storing short segments of viral DNA that can be recognized during future encounters. The Cas9 protein acts as molecular scissors, guided by a short RNA sequence to cut DNA at a precise location. Jennifer Doudna and Emmanuelle Charpentier demonstrated that this system could be reprogrammed to cut any DNA sequence by changing the guide RNA, earning them the 2020 Nobel Prize in Chemistry. The technology has since been used to correct genetic mutations in human cells, engineer disease-resistant crops, create gene drives in mosquitoes to combat malaria, and develop new diagnostic tests for infectious diseases. Ethical debates continue about the boundaries of human germline editing, particularly after a Chinese researcher controversially edited the genomes of twin embryos in 2018.",

    # Technology
    "Modern semiconductor fabrication operates at scales that challenge intuition. The transistors in a contemporary processor are measured in nanometers, where a nanometer is one billionth of a meter, roughly the width of a few atoms. A single chip may contain over 100 billion transistors packed into an area smaller than a fingernail. Manufacturing these devices requires extreme ultraviolet lithography, which uses light with a wavelength of just 13.5 nanometers to pattern features onto silicon wafers. The light source works by firing a high-powered laser at tiny tin droplets, each about 25 micrometers in diameter, dropped at a rate of 50,000 per second. The resulting plasma emits EUV radiation that is collected by specialized mirrors and focused through a series of optics. Each mirror must be polished to sub-atomic smoothness, with surface irregularities measuring less than the diameter of a single atom. The machines that perform this process cost roughly 400 million dollars each and weigh over 180 tons.",

    # Economics
    "The relationship between inflation and unemployment, described by the Phillips curve, has been one of the most debated topics in macroeconomics since A.W. Phillips published his empirical finding in 1958. The original observation was straightforward: when unemployment was low, wages tended to rise faster, pushing up prices. Policymakers initially interpreted this as a stable tradeoff, suggesting they could choose their preferred combination of inflation and unemployment. This view was challenged in the late 1960s by Milton Friedman and Edmund Phelps, who argued that any attempt to maintain unemployment below its natural rate would lead to accelerating inflation as workers adjusted their expectations. The stagflation of the 1970s, with simultaneous high inflation and high unemployment, appeared to confirm their critique. More recent research has noted that the Phillips curve has flattened considerably since the 1990s, with large changes in unemployment producing only modest effects on inflation, a puzzle that central bankers continue to grapple with.",

    # Literature and Philosophy
    "The problem of consciousness remains one of the deepest unsolved questions in philosophy and neuroscience. David Chalmers formalized the distinction between easy and hard problems of consciousness in 1995. The easy problems, though scientifically challenging, involve explaining cognitive functions like attention, memory consolidation, and behavioral responses to stimuli. The hard problem asks why and how physical processes in the brain give rise to subjective experience at all. When you see the color red, there is something it is like to have that experience, a qualitative character that seems to resist explanation in purely physical terms. Some philosophers, like Daniel Dennett, argue that the hard problem is an illusion generated by our confused intuitions about the nature of mind. Others, like Chalmers himself, suggest that consciousness may be a fundamental feature of reality, not reducible to physical processes. Neuroscientific approaches such as integrated information theory and global workspace theory attempt to bridge the gap by identifying neural correlates of consciousness, but whether correlation implies explanation remains contested.",

    # Engineering
    "Bridge design illustrates the tension between efficiency and redundancy in structural engineering. A simply supported beam bridge is the most intuitive form, with the deck resting on supports at each end, but it becomes impractical for spans beyond about 80 meters because the bending moments grow with the square of the span length. Truss bridges address this by distributing forces through a network of triangular elements, where each member carries either pure tension or pure compression. Suspension bridges can cross much greater distances by hanging the deck from cables draped over tall towers, with the main cables transferring the load to massive anchorages. The Golden Gate Bridge spans 1,280 meters with cables containing 80,000 miles of wire. Cable-stayed bridges use a different approach, running cables directly from the towers to the deck in a fan or harp pattern. Modern computational methods allow engineers to optimize material usage through finite element analysis, but experienced designers know that structural elegance and constructability matter as much as mathematical optimality.",
]


EVALUATE_CODE = '''
"""Evaluate quantization quality on Qwen3-4B.

Loads reference and solution models SEQUENTIALLY to fit in 24GB VRAM.
Computes perplexity on held-out English passages loaded from a HIDDEN file
outside the workspace (agents cannot read the eval texts).

Quality = exp(-max(0, delta_ppl_norm)) where delta_ppl_norm = (sol_ppl - ref_ppl) / ref_ppl
Fitness = quality * log2(compression_ratio)
"""
import gc
import importlib.util
import json
import math
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"
REF_BYTES = 8_044_936_192

# Eval texts are loaded from a hidden file outside the workspace.
# Agents CANNOT read this file -- it is not in their sandbox.
_EVAL_TEXTS_PATH = os.environ.get("CRUCIBLE_EVAL_TEXTS", "")
if not _EVAL_TEXTS_PATH or not os.path.exists(_EVAL_TEXTS_PATH):
    print(json.dumps({"fitness": 0.0, "error": "Hidden eval texts not found"}))
    sys.exit(0)
with open(_EVAL_TEXTS_PATH) as _f:
    EVAL_TEXTS = json.load(_f)

MAX_SEQ_LEN = 512  # Truncate to this length for evaluation


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_perplexity(model, input_ids_list, device):
    """Compute perplexity across multiple sequences."""
    total_loss = 0.0
    total_tokens = 0
    for input_ids in input_ids_list:
        ids = input_ids[:MAX_SEQ_LEN].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(ids)
            # Handle both raw logits and model output objects
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            # Next-token prediction: logits[:-1] predicts tokens[1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                reduction="sum",
            )
        total_loss += loss.item()
        total_tokens += shift_labels.numel()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


try:
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Tokenize eval texts
    input_ids_list = []
    for text in EVAL_TEXTS:
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)[0]
        if len(ids) > 2:  # Skip degenerate sequences
            input_ids_list.append(ids)

    if not input_ids_list:
        print(json.dumps({"fitness": 0.0, "error": "No valid eval sequences"}))
        sys.exit(0)

    total_eval_tokens = sum(len(ids) for ids in input_ids_list)

    # ---- Phase 1: Reference model perplexity ----
    ref = load_module("reference", "reference.py")
    ref_model = ref.load_model(device=device)
    ref_ppl = compute_perplexity(ref_model, input_ids_list, device)
    del ref_model, ref
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Phase 2: Solution model ----
    sol = load_module("solution", "solution.py")

    if not hasattr(sol, "get_compressed_size"):
        print(json.dumps({
            "fitness": 0.0,
            "error": "Solution must define get_compressed_size() -> int"
        }))
        sys.exit(0)

    # Load solution model
    if hasattr(sol, "load_model"):
        sol_model = sol.load_model(device=device)
    elif hasattr(sol, "Model"):
        sol_model = sol.Model()
        if hasattr(sol_model, "to"):
            sol_model = sol_model.to(device)
        sol_model.eval()
    else:
        print(json.dumps({
            "fitness": 0.0,
            "error": "Solution must define load_model(device) or Model class"
        }))
        sys.exit(0)

    sol_bytes = sol.get_compressed_size()

    if not isinstance(sol_bytes, (int, float)) or sol_bytes <= 0:
        print(json.dumps({
            "fitness": 0.0,
            "error": f"get_compressed_size() must return positive number, got {sol_bytes}"
        }))
        sys.exit(0)

    sol_bytes = int(sol_bytes)

    if sol_bytes >= REF_BYTES:
        print(json.dumps({
            "fitness": 0.0,
            "error": f"No compression: sol_bytes={sol_bytes} >= ref_bytes={REF_BYTES}",
        }))
        sys.exit(0)

    compression_ratio = REF_BYTES / sol_bytes

    # Sanity: compression > 32x is suspicious for a 4B model
    if compression_ratio > 32:
        print(json.dumps({
            "fitness": 0.0,
            "error": f"Compression ratio {compression_ratio:.1f}x exceeds 32x cap -- likely dishonest size claim",
        }))
        sys.exit(0)

    sol_ppl = compute_perplexity(sol_model, input_ids_list, device)

    # Check for degenerate output
    if math.isnan(sol_ppl) or math.isinf(sol_ppl) or sol_ppl > 1e6:
        print(json.dumps({
            "fitness": 0.0,
            "error": f"Degenerate perplexity: {sol_ppl}",
            "ref_ppl": round(ref_ppl, 4),
        }))
        sys.exit(0)

    # Sanity check: a quantized model CANNOT have meaningfully lower perplexity
    # than the fp16 reference. If sol_ppl < ref_ppl * 0.8, something is wrong
    # (memorized eval data, hardcoded outputs, logit manipulation, etc.)
    if sol_ppl < ref_ppl * 0.8:
        print(json.dumps({
            "fitness": 0.0,
            "error": f"Suspiciously low perplexity: sol_ppl={sol_ppl:.4f} < 0.8*ref_ppl={ref_ppl*0.8:.4f}. "
                     f"Quantization cannot improve model quality this much -- likely metric gaming.",
            "ref_ppl": round(ref_ppl, 4),
            "sol_ppl": round(sol_ppl, 4),
        }))
        sys.exit(0)

    # Quality = exp(-max(0, normalized ppl increase))
    # Normalized by ref_ppl so the scale is relative, not absolute
    delta_ppl_norm = (sol_ppl - ref_ppl) / ref_ppl
    quality = math.exp(-max(0.0, delta_ppl_norm))
    quality = min(quality, 1.0)  # Cap at 1.0

    # Fitness = quality * log2(compression_ratio)
    fitness = quality * math.log2(max(compression_ratio, 1.01))

    del sol_model, sol
    gc.collect()
    torch.cuda.empty_cache()

    print(json.dumps({
        "fitness": round(fitness, 6),
        "quality": round(quality, 6),
        "compression_ratio": round(compression_ratio, 4),
        "ref_ppl": round(ref_ppl, 4),
        "sol_ppl": round(sol_ppl, 4),
        "delta_ppl_pct": round(delta_ppl_norm * 100, 2),
        "ref_bytes": REF_BYTES,
        "sol_bytes": sol_bytes,
        "eval_tokens": total_eval_tokens,
    }))

except Exception as e:
    import traceback
    print(json.dumps({
        "fitness": 0.0,
        "error": str(e),
        "traceback": traceback.format_exc()[:2000],
    }))
'''


class QuantizationProtocol(Protocol):
    """Quantize Qwen3-4B while preserving quality.

    Fitness = quality * log2(compression_ratio).
    Quality = exp(-max(0, (sol_ppl - ref_ppl) / ref_ppl)).
    Perplexity measured on held-out English passages.
    """

    def __init__(self, **kwargs):
        pass

    @property
    def name(self) -> str:
        return "quantization"

    @property
    def fitness_key(self) -> str:
        return "fitness"

    @property
    def fitness_direction(self) -> str:
        return "max"

    def setup_workspace(self, sandbox: Sandbox, round_num: int, prior_best: Optional[str]) -> None:
        sandbox.write_file("reference.py", REFERENCE_CODE)
        sandbox.write_file("_evaluate.py", EVALUATE_CODE)

        # Write eval texts to a HIDDEN location outside the sandbox workspace.
        # Agents cannot read this file -- it's passed via env var at eval time.
        self._eval_texts_path = tempfile.mktemp(prefix="crucible_eval_", suffix=".json")
        with open(self._eval_texts_path, "w") as f:
            json.dump(EVAL_TEXTS, f)

        if prior_best:
            sandbox.write_file("prior_best.py", prior_best)

    def get_system_prompt(self) -> str:
        return f"""You are a quantization researcher competing in a tournament.

Your task: quantize Qwen/Qwen3-4B (4.02B params, 8.04 GB fp16) while preserving
text generation quality. The model is a dense transformer for text-generation.

TARGET GPU: NVIDIA RTX 3090 (24GB VRAM, CUDA)

MODEL ARCHITECTURE:
- Hidden: 2560, Layers: 36, Heads: 32 (8 KV heads, GQA), Intermediate: 9728
- Vocab: 151,936, Context: 40,960
- SwiGLU activation, RoPE positional encoding
- FP16 reference size: 8,044,936,192 bytes

YOUR SOLUTION (solution.py) MUST define:
1. load_model(device="cuda") -> model  OR  a Model class
   - Must accept input_ids tensor and return logits (or object with .logits)
   - Same interface as: model(input_ids) -> logits of shape (B, T, vocab_size)
2. get_compressed_size() -> int
   - Actual compressed model size in bytes
   - Must be LESS than 8,044,936,192 (the fp16 reference)
   - Must be HONEST -- reflects real bits used, not a fabricated number
   - Compression ratio capped at 32x (anything higher is rejected)

EVALUATION:
- Perplexity on held-out English passages (science, history, math, biology, tech, etc.)
- Quality = exp(-max(0, (sol_ppl - ref_ppl) / ref_ppl))
  Even a 10% perplexity increase gives quality = exp(-0.1) = 0.905
  A 50% increase gives quality = exp(-0.5) = 0.607
- Fitness = quality * log2(compression_ratio)
- Models are loaded SEQUENTIALLY (ref first, then solution) to fit in 24GB

APPROACHES TO CONSIDER:
- GPTQ: second-order weight quantization, state-of-the-art for 4-bit
  pip install auto-gptq optimum
- AWQ: activation-aware weight quantization, fast inference
  pip install autoawq
- bitsandbytes: NF4/INT8 quantization, easy to use
  pip install bitsandbytes
- Custom schemes: per-channel scales, mixed precision across layers,
  outlier-aware quantization, entropy coding, pruning + quantization
- Sensitivity analysis: attention layers (especially Q/K) are typically more
  sensitive than MLP layers. Embeddings and LM head are often robust to
  lower precision.

ANALYSIS STEPS (do these first):
1. Read reference.py -- understand architecture and helpers
2. Run: python -c "import reference; import json; print(json.dumps(reference.get_model_config(), indent=2))"
3. Analyze per-layer weight distributions if time permits
4. Choose quantization strategy based on target compression ratio
5. Implement in solution.py
6. Test: python _evaluate.py
7. Submit when satisfied

CRITICAL: Before submitting, make sure your solution works end-to-end:
1. Run your quantization script to generate compressed weight files
2. Verify load_model() works: python3 -c "import solution; m = solution.load_model(); print('OK')"
3. Verify get_compressed_size() returns a valid number
4. You CANNOT run _evaluate.py directly -- the eval texts are hidden and loaded from
   a secret path at evaluation time. Focus on making load_model() and get_compressed_size() correct.
5. Evaluation will load the fp16 reference and your quantized model SEQUENTIALLY (not
   simultaneously) and measure perplexity on held-out English passages you have never seen.

The evaluation loads your solution fresh -- if your quantized weights aren't saved to disk,
they'll need to be regenerated during evaluation (which may timeout at 600s).

ANTI-GAMING: The eval texts are NOT in _evaluate.py or anywhere in your workspace. Any
attempt to overfit to specific text (exact token caching, logit temperature tuning on known
texts, etc.) will fail because you don't know the eval distribution. A quantized model that
achieves perplexity more than 20% below the fp16 reference will be flagged as gaming and
score 0.

TOOLS: read_file, write_file, edit_file, bash, submit
Do NOT modify reference.py or _evaluate.py.
"""

    def get_initial_message(
        self, round_num: int, prior_best: Optional[str], prior_fitness: Optional[float]
    ) -> str:
        msg = f"Round {round_num} of the quantization tournament.\n\n"
        msg += "Target: Qwen/Qwen3-4B (4.02B params, 8.04 GB fp16, dense text-generation).\n"
        msg += "Quantize it while preserving perplexity on held-out English text.\n"
        msg += "Fitness = quality * log2(compression_ratio).\n\n"

        if prior_best and prior_fitness:
            msg += f"""The current best solution achieved fitness={prior_fitness:.4f}.
It is saved as prior_best.py. You must BEAT this score.

Read it, understand the approach, then find improvements:
- More aggressive quantization for insensitive layers (MLP, embeddings)
- Better calibration data or Hessian estimation
- Mixed-precision: 3-bit for robust layers, 4-bit for sensitive ones
- Outlier channels at higher precision (SqueezeLLM-style)
- Group quantization with smaller group sizes for better accuracy
- Novel approaches: pruning + quantization, weight clustering, etc.
"""
        else:
            msg += """First round -- no prior solutions.

Recommended starting approach:
1. Read reference.py to understand the model interface
2. Try GPTQ 4-bit as a strong baseline:
   pip install auto-gptq optimum
   This gives ~4x compression with minimal quality loss
3. If that works, explore: 3-bit GPTQ, mixed-precision, AWQ, custom schemes
4. Test with: python _evaluate.py
5. Submit when ready
"""
        return msg

    def evaluate(self, sandbox: Sandbox) -> Dict[str, Any]:
        # Pass hidden eval texts path via env var -- agents can't access this file
        eval_cmd = f"CRUCIBLE_EVAL_TEXTS={self._eval_texts_path} python3 _evaluate.py"
        result = sandbox.run_command(eval_cmd, timeout=600)
        stdout = result["stdout"].strip()
        stderr = result.get("stderr", "").strip()
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        # Include stderr in error for debugging
        err_detail = stdout[:300] if stdout else "(no stdout)"
        if stderr:
            err_detail += f" | stderr: {stderr[:300]}"
        return {"fitness": 0.0, "error": f"Parse failed: {err_detail}"}
