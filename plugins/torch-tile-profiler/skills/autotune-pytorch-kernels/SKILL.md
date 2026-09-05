---
name: autotune-pytorch-kernels
description: Benchmark and rank PyTorch eager, torch.compile modes, attention SDPA, channels-last convolution, and Triton matmul tile configurations using synchronized measurements. Use when selecting kernel configurations, sweeping tile sizes, reducing PyTorch workload latency, or quantifying optimization speedups on a specific accelerator.
---

# Autotune PyTorch Kernels

## Workflow

1. Record the exact device, dtype, and dimensions. Do not compare candidates with different shapes or precision.
2. Call `autotune_workload`. For matmul, keep Triton enabled on CUDA and provide tile sizes; for attention and convolution, let the tool add fused SDPA and channels-last candidates.
3. Select only from successful measured candidates. Treat failed candidates as unsupported configurations, not slow results.
4. Report both latency reduction, `(baseline - best) / baseline`, and throughput gain, `(best throughput / baseline throughput) - 1`; they are not the same percentage.
5. Preserve the JSON/CSV result artifacts so reported speedups can be recomputed from raw timings.

## Candidate strategy

- For a compute-bound matmul, sweep `torch.compile` modes plus Triton block sizes, warps, and stages.
- For a compute-bound convolution, compare contiguous and channels-last layouts with compiled variants.
- For memory-bound attention, prioritize fused scaled-dot-product attention to avoid materializing the score matrix.
- Keep eager execution as the baseline and as the winner when no candidate beats it.

## CLI fallback

If the MCP server is unavailable, run:

```bash
torch-tile-profiler autotune --workload matmul --device cuda --dtype float16 \
  --m 4096 --n 4096 --k 4096 --tile-sizes 32 64 128 \
  --warmup 20 --iterations 100 --json reports/matmul-autotune.json
```
