# PyTorch Compute Tile Utilization Profiler

An open-source PyTorch profiling tool that estimates FLOPs, memory traffic,
arithmetic intensity, and compute-vs-memory bottlenecks for tensor workloads.
It wraps `torch.profiler`, runs repeatable workloads, and exports runtime tables
as JSON or CSV.

![Runtime table screenshot](docs/screenshots/runtime-table.svg)

## Features

- Profiles matrix multiplication, convolution, and transformer-style attention.
- Uses `torch.profiler` for CPU/CUDA timing and operator traces.
- Estimates FLOPs, memory traffic, arithmetic intensity, and bottleneck class.
- Compares PyTorch eager mode and `torch.compile` when available.
- Sweeps matrix tile sizes to show how shape choices affect utilization.
- Exports machine-readable JSON and CSV reports.
- Includes a roofline-style summary using configurable peak FLOP/s and bandwidth.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[torch,dev]"

torch-tile-profiler profile --workload matmul --device cuda --mode eager --json reports/matmul.json --csv reports/matmul.csv
```

CPU-only machines work too:

```bash
torch-tile-profiler profile --workload conv2d --device cpu --mode eager
```

Compare eager, compiled PyTorch, and the optional Triton tile kernel:

```bash
python3 -m pip install -e ".[torch,triton]"
torch-tile-profiler compare --workload matmul --device cuda --modes eager compile triton
```

Run a tile sweep:

```bash
torch-tile-profiler sweep --device cuda --m 4096 --n 4096 --k 4096 --tile-sizes 16 32 64 128
```

## Measured NVIDIA T4 Results

These results were measured on Google Compute Engine with one NVIDIA Tesla T4,
an `n1-standard-4` host, PyTorch 2.9.1+cu129, FP16 tensors, 20 warmup
iterations, and 100 measured iterations. The roofline configuration used 65
peak TFLOP/s and 320 GB/s of memory bandwidth.

Runtime is measured from real CUDA execution. Logical throughput combines the
measured runtime with the estimated workload FLOPs. Arithmetic intensity and
bottleneck classification are analytical predictions, not GPU hardware
counters.

### Workload Profiles

| Workload | Mode | Device | Measured time (ms) | Logical GFLOP/s | Estimated AI (F/B) | Predicted bottleneck |
| --- | --- | --- | ---: | ---: | ---: | --- |
| matmul | eager | cuda | 5.59 | 24567.6 | 1365.33 | compute-bound |
| conv2d | eager | cuda | 1.35 | 10926.5 | 382.54 | compute-bound |
| attention | eager | cuda | 7.89 | 4458.8 | 104.80 | memory-bound |

### Matmul Implementation Comparison

| Mode | Device | Measured time (ms) | Logical GFLOP/s | Estimated AI (F/B) | Predicted bottleneck |
| --- | --- | ---: | ---: | ---: | --- |
| eager | cuda | 6.07 | 22647.7 | 1365.33 | compute-bound |
| compile | cuda | 5.88 | 23365.4 | 1365.33 | compute-bound |
| triton | cuda | 67.19 | 2045.6 | 1365.33 | compute-bound |

The comparison shows that eager and compiled PyTorch use highly optimized
matrix multiplication implementations on the T4. The project's educational,
fixed-block Triton kernel is not yet autotuned and is substantially slower.

## How Bottlenecks Are Classified

For each workload, the profiler computes:

```text
arithmetic_intensity = estimated_flops / estimated_memory_bytes
roofline_threshold = peak_flops_per_second / peak_bandwidth_bytes_per_second
```

If arithmetic intensity is below the threshold, the workload is classified as
memory-bound. Otherwise, it is compute-bound. Defaults are conservative and can
be overridden:

```bash
torch-tile-profiler profile --peak-tflops 82.6 --bandwidth-gbps 1008
```

## Project Layout

```text
src/torch_tile_profiler/
  cli.py          # command-line interface
  estimator.py    # FLOP/memory/arithmetic intensity formulas
  profiler.py     # torch.profiler orchestration
  workloads.py    # matmul, conv2d, attention workload definitions
  reports.py      # JSON/CSV export and terminal tables
tests/
  test_estimator.py
```
