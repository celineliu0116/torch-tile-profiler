---
name: diagnose-pytorch-gpu
description: Profile PyTorch matmul, convolution, and attention workloads; apply roofline analysis; compare eager, torch.compile, optimized layouts or fusions, and Triton; and produce evidence-backed GPU performance reports. Use for GPU bottleneck diagnosis, performance regression analysis, or requests to explain whether a workload is compute- or memory-bound.
---

# Diagnose PyTorch GPU

## Workflow

1. Confirm the target device, dtype, workload shapes, warmup count, measured iterations, peak TFLOP/s, and memory bandwidth. Use the user's values when supplied.
2. Call the `diagnose_pytorch` MCP tool for the full suite. Use `profile_workload` for a single configuration.
3. Inspect `diagnosis.json`, treating runtime and derived throughput as measured and FLOPs, memory traffic, arithmetic intensity, and bottleneck class as analytical estimates.
4. Report the baseline, fastest measured configuration, latency reduction, throughput gain, roofline classification, and any failed candidates.
5. Report measured results only for the accelerator and workload shapes that were actually benchmarked.

## Guardrails

- Keep shapes, dtype, accelerator name, PyTorch version, warmups, and iterations with every reported result.
- Use CUDA-event timings for CUDA comparisons and compare identical workload shapes.
- Describe roofline classifications as model predictions, not hardware-counter measurements.
- If CUDA is unavailable, run `tile_size_sweep` for analytical insight and clearly say GPU speedups were not measured.
- Prefer at least 20 warmups and 100 iterations for externally reported numbers.

## CLI fallback

If the MCP server is unavailable, run:

```bash
torch-tile-profiler diagnose --device cuda --dtype float16 \
  --peak-tflops 65 --bandwidth-gbps 320 \
  --warmup 20 --iterations 100 --output-dir reports/t4
```
