"""Kernel Evolution Protocol: agents write optimized GPU kernels, tournament selects fastest."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.protocols.base import Protocol
from src.sandbox import Sandbox

# Default problems directory (KernelBench-v3 problems)
PROBLEMS_DIR = Path(__file__).parent.parent.parent / "problems"
KERNELBENCH_PROBLEMS = Path.home() / "cuda" / "KernelBench-v3" / "problems"


def _discover_problems(problems_dir: Path, levels: Optional[List[int]] = None) -> List[Path]:
    """Find all problem .py files in the given directory."""
    problems = []
    level_dirs = []
    if levels:
        for lvl in levels:
            d = problems_dir / f"level{lvl}"
            if d.exists():
                level_dirs.append(d)
    else:
        for d in sorted(problems_dir.iterdir()):
            if d.is_dir() and d.name.startswith("level"):
                level_dirs.append(d)
    for d in level_dirs:
        for f in sorted(d.glob("*.py")):
            problems.append(f)
    return problems


# ---------------------------------------------------------------------------
# Tamper-proof benchmark
#
# Reference and solution are timed in SEPARATE processes so the solution
# cannot monkey-patch the reference, intercept CUDA events, or otherwise
# manipulate the timing of the baseline.
#
# Sanity checks:
#   - ref_ms must be in [0.001, 1000] ms
#   - sol_ms must be > 0
#   - speedup is capped at 500x (anything above is clearly gaming)
#   - 30 runs, drop top/bottom 5, median of remaining 20
# ---------------------------------------------------------------------------

# Helper script run in a subprocess to time a single module.
_TIMER_SUBPROCESS = r'''
import importlib.util, json, statistics, sys, torch

device = torch.device("cuda:0")

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

module_path = sys.argv[1]  # path to .py file
mod = load_module("target", module_path)

ref = load_module("reference", "reference.py")
init_inputs = ref.get_init_inputs()
model = mod.Model(*init_inputs).to(device).eval()

# Try to share weights from reference
try:
    ref_model = ref.Model(*init_inputs).to(device).eval()
    model.load_state_dict(ref_model.state_dict(), strict=False)
    del ref_model
except Exception:
    pass

inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in ref.get_inputs()]

# Warmup
for _ in range(10):
    with torch.no_grad():
        model(*inputs)
torch.cuda.synchronize()

# Timed runs -- 30 iterations
times = []
for _ in range(30):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        model(*inputs)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

# Drop top/bottom 5 outliers, take median of remaining 20
times.sort()
trimmed = times[5:25]
median_ms = statistics.median(trimmed)

print(json.dumps({"median_ms": median_ms}))
'''

BENCHMARK_TEMPLATE = '''
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
import torch

device = torch.device("cuda:0")

TIMER_CODE = __TIMER_CODE_PLACEHOLDER__

MAX_SPEEDUP = 500.0
REF_MS_MIN = 0.001
REF_MS_MAX = 1000.0

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def time_module_in_subprocess(module_path: str) -> float:
    """Run the timer script in a fresh process and return median ms."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(TIMER_CODE)
        timer_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, timer_path, module_path],
            capture_output=True, text=True, timeout=90,
            cwd=os.getcwd(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Timer subprocess failed: {result.stderr[:500]}")
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                data = json.loads(line)
                return data["median_ms"]
        raise RuntimeError(f"No JSON output from timer: {result.stdout[:500]}")
    finally:
        os.unlink(timer_path)

try:
    ref = load_module("reference", "reference.py")
    sol = load_module("solution", "solution.py")

    # Build models for correctness check only
    init_inputs = ref.get_init_inputs()
    ref_model = ref.Model(*init_inputs).to(device).eval()
    sol_model = sol.Model(*init_inputs).to(device).eval()

    # Share weights
    try:
        sol_model.load_state_dict(ref_model.state_dict(), strict=False)
    except Exception:
        pass

    # Correctness check (3 seeds) -- tighter tolerances
    correct = True
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in ref.get_inputs()]
        with torch.no_grad():
            ref_out = ref_model(*inputs)
            sol_out = sol_model(*inputs)
        if isinstance(ref_out, tuple):
            ref_out = ref_out[0]
        if isinstance(sol_out, tuple):
            sol_out = sol_out[0]
        if ref_out.shape != sol_out.shape:
            correct = False
            break
        if torch.isnan(sol_out).any() or torch.isinf(sol_out).any():
            correct = False
            break
        if not torch.allclose(ref_out, sol_out, atol=1e-3, rtol=1e-3):
            max_diff = (ref_out - sol_out).abs().max().item()
            if max_diff > 0.05:
                correct = False
                break

    if not correct:
        print(json.dumps({"correct": False, "speedup": 0.0, "error": "correctness_failed"}))
        sys.exit(0)

    # Free correctness-check models before benchmark subprocesses
    del ref_model, sol_model
    torch.cuda.empty_cache()

    # ---- Tamper-proof benchmark: separate processes ----
    ref_ms = time_module_in_subprocess(os.path.abspath("reference.py"))
    sol_ms = time_module_in_subprocess(os.path.abspath("solution.py"))

    # Sanity checks
    if not (REF_MS_MIN <= ref_ms <= REF_MS_MAX):
        print(json.dumps({
            "correct": False, "speedup": 0.0,
            "error": f"ref_ms={ref_ms:.6f} outside sane range [{REF_MS_MIN}, {REF_MS_MAX}] -- timing manipulation detected",
        }))
        sys.exit(0)

    if sol_ms <= 0:
        print(json.dumps({
            "correct": False, "speedup": 0.0,
            "error": f"sol_ms={sol_ms:.6f} <= 0 -- timing manipulation detected",
        }))
        sys.exit(0)

    speedup = ref_ms / sol_ms
    if speedup > MAX_SPEEDUP:
        speedup = 0.0
        print(json.dumps({
            "correct": False, "speedup": 0.0,
            "error": f"Reported speedup {ref_ms/sol_ms:.1f}x exceeds {MAX_SPEEDUP}x cap -- score zeroed",
        }))
        sys.exit(0)

    print(json.dumps({
        "correct": True,
        "speedup": round(speedup, 4),
        "ref_ms": round(ref_ms, 4),
        "sol_ms": round(sol_ms, 4),
    }))

except Exception as e:
    print(json.dumps({"correct": False, "speedup": 0.0, "error": str(e)}))
'''


class KernelEvolutionProtocol(Protocol):
    """Adversarial kernel optimization tournament.

    Agents write Triton/CUDA kernels to beat PyTorch reference implementations.
    Fitness = speedup over PyTorch baseline.
    """

    def __init__(
        self,
        problem_path: Optional[str] = None,
        levels: Optional[List[int]] = None,
        **kwargs,
    ):
        # Pick a specific problem or random from available
        if problem_path:
            self._problem_path = Path(problem_path)
        else:
            # Prefer KernelBench (full set), fall back to local problems
            if KERNELBENCH_PROBLEMS.exists():
                problems_dir = KERNELBENCH_PROBLEMS
            elif PROBLEMS_DIR.exists():
                problems_dir = PROBLEMS_DIR
            else:
                raise RuntimeError(
                    f"No problems directory found. Checked:\n"
                    f"  {KERNELBENCH_PROBLEMS}\n  {PROBLEMS_DIR}"
                )
            problems = _discover_problems(problems_dir, levels or [1, 2, 3])
            if not problems:
                raise RuntimeError(f"No problems found in {problems_dir}")
            self._problem_path = random.choice(problems)

        self._problem_code = self._problem_path.read_text()
        self._problem_name = self._problem_path.stem

    @property
    def name(self) -> str:
        return "kernel"

    @property
    def fitness_key(self) -> str:
        return "speedup"

    @property
    def fitness_direction(self) -> str:
        return "max"

    def setup_workspace(self, sandbox: Sandbox, round_num: int, prior_best: Optional[str]) -> None:
        sandbox.write_file("reference.py", self._problem_code)
        # Inject the timer subprocess code into the benchmark template
        benchmark = BENCHMARK_TEMPLATE.replace(
            "__TIMER_CODE_PLACEHOLDER__", repr(_TIMER_SUBPROCESS)
        )
        sandbox.write_file("_benchmark.py", benchmark)

        # If there's a prior best solution, include it
        if prior_best:
            sandbox.write_file("prior_best.py", prior_best)

    def get_system_prompt(self) -> str:
        return f"""You are a GPU kernel optimization expert competing in a tournament.

Your task: write an optimized GPU kernel (Triton or CUDA via torch.utils.cpp_extension.load_inline)
that is FASTER than the PyTorch reference implementation while producing correct results.

RULES:
- Write your solution to solution.py
- Your solution must define a Model class with the same interface as reference.py
- You may use: triton, torch.utils.cpp_extension.load_inline (for CUDA C++), or pure PyTorch optimizations
- Do NOT use: flash_attn, xformers, or other external kernel libraries
- Do NOT modify reference.py or _benchmark.py
- Test correctness before submitting
- When ready, use the submit tool

ANTI-TAMPERING NOTICE:
- The benchmark runs reference and solution in SEPARATE PROCESSES.
- Timing validated: ref_ms in [0.001, 1000], sol_ms > 0. Speedup capped at 500x.
- Gaming = score 0. Correctness: atol=1e-3, rtol=1e-3, max abs diff < 0.05.

TARGET GPU: NVIDIA RTX 3090 (GA102, SM86, Ampere)
- 82 SMs, 128 CUDA cores/SM, 10496 total
- DRAM: 24GB GDDR6X, 384-bit bus, 936 GB/s practical bandwidth
- L2: 6MB shared across all SMs, 128B lines (4x32B sectors)
- L1/SM: 128KB configurable (48KB L1 + 48KB smem default), 128B lines
- Registers/SM: 65536 (256KB), max 1536 threads/SM (48 warps)
- Constant memory: 64KB, broadcast to all threads via constant cache
- Tensor cores: 3rd gen WMMA (warp-level, needs M/N/K multiples of 16)
- FP32 peak: 35.6 TFLOPS, TF32 peak: 142 TFLOPS

OPTIMIZATION METHODOLOGY (follow this order):
1. SPEED OF LIGHT FIRST: Before writing any code, compute the DRAM bandwidth floor:
   floor_us = (input_bytes + output_bytes) / 936e9 * 1e6
   Compute arithmetic intensity = FLOPs / total_bytes.
   If AI < 38 FLOP/byte: MEMORY-BOUND. Optimize for fewer bytes moved, not faster math.
   If AI > 38 FLOP/byte: COMPUTE-BOUND. Use tensor cores, optimize instruction throughput.

2. FOR MEMORY-BOUND KERNELS (most elementwise, small-channel conv, pooling, normalization):
   - FUSE ALL OPS into a single kernel. Each eliminated intermediate tensor saves a full
     DRAM round-trip. This is the #1 optimization -- nothing else matters until you fuse.
   - Consider REGISTER-ONLY computation (no shared memory) when working set per thread is
     small (<60 floats). Eliminates __syncthreads() overhead. L1 cache handles cross-thread
     reuse automatically. We measured: register-only beats shared-memory by 16% on conv+pool.
   - Use __constant__ memory for weights when they fit in 64KB. Free broadcast, no bank conflicts.
   - Process all output channels per thread when OC is small (<=32). This reads input once
     and reuses across all OC -- critical for amortizing memory traffic.
   - Things that DON'T help when memory-bound: fast math approximations (tanh, exp -- latency
     is hidden by memory stalls), bank conflict padding, NHWC layout for IC<16, persistent
     kernels with fewer blocks than SMs.

3. FOR COMPUTE-BOUND KERNELS (large GEMM, attention with long sequences):
   - Implicit GEMM for convolutions: treat as (N*H*W, IC*KH*KW) @ (IC*KH*KW, OC) matrix
     multiply. Amortizes im2col gather across output channels.
   - Tensor cores via WMMA when K >= 16. Pad if needed but beware wasted FLOPs.
   - Tile sizes: 128x128x32 or 256x128x32 for GEMM on Ampere.
   - Double-buffer with cp.async (PTX: cp.async.ca.shared.global) to overlap load and compute.

4. PROFILING: Use torch.profiler with ProfilerActivity.CUDA for actual kernel duration.
   CUDA event timing UNDERCOUNTS when kernels pipeline (back-to-back launches overlap).
   Never trust sub-10us event measurements. The profiler gives true wall-clock per kernel.

5. VERIFY with roofline: measured_time / floor_time = your efficiency ratio.
   1.0x = perfect. 1.3-1.6x = good. >2x = significant room for improvement.

PROBLEM: {self._problem_name}
"""

    def get_initial_message(
        self, round_num: int, prior_best: Optional[str], prior_fitness: Optional[float]
    ) -> str:
        msg = f"""Round {round_num} of the kernel optimization tournament.

Problem: {self._problem_name}

The reference implementation is in reference.py. Read it first to understand what operation to optimize.

"""
        if prior_best and prior_fitness:
            msg += f"""The current best solution achieved {prior_fitness:.4f}x speedup over PyTorch.
It is saved as prior_best.py in your workspace. You must BEAT this speedup.

STEPS:
1. Read reference.py -- understand shapes, dtypes, operation
2. Compute speed-of-light: input_bytes + output_bytes, arithmetic intensity, DRAM floor
3. Read prior_best.py -- understand current approach
4. Identify the bottleneck: is prior_best memory-bound or compute-bound?
5. Write a kernel that either:
   a. Moves fewer bytes (eliminate redundant loads, better tiling, register-only)
   b. Computes faster (tensor cores, better instruction mix, less divergence)
   c. Uses a fundamentally different algorithm (implicit GEMM vs direct conv, Winograd, etc.)
6. Verify correctness, then submit

"""
        else:
            msg += """This is the first round -- no prior solutions exist yet.

STEPS:
1. Read reference.py -- understand the operation, shapes, dtypes
2. Compute speed-of-light: calculate total bytes moved and DRAM floor time
3. Determine if memory-bound or compute-bound (arithmetic intensity vs 38 FLOP/byte ridge)
4. Choose approach:
   - Memory-bound: fuse all ops, minimize intermediate tensors, register-only if working set small
   - Compute-bound: implicit GEMM, tensor cores, optimal tiling
5. Write your kernel in solution.py
6. Verify correctness, then submit
"""
        return msg

    def evaluate(self, sandbox: Sandbox) -> Dict[str, Any]:
        result = sandbox.run_command("python3 _benchmark.py", timeout=180)
        stdout = result["stdout"].strip()

        # Parse the last line as JSON
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        return {
            "correct": False,
            "speedup": 0.0,
            "error": f"Could not parse benchmark output: {stdout[:500]}",
        }
