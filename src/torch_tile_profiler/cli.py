from __future__ import annotations

import argparse

from .estimator import HardwareRoofline, matmul_estimate, tile_utilization
from .profiler import profile_workload
from .reports import print_table, write_csv, write_json
from .workloads import get_workload


def _roofline(args: argparse.Namespace) -> HardwareRoofline:
    return HardwareRoofline(peak_tflops=args.peak_tflops, bandwidth_gbps=args.bandwidth_gbps)


def _export(results: list[object], args: argparse.Namespace) -> None:
    print_table(results)
    if args.json:
        write_json(results, args.json)
    if args.csv:
        write_csv(results, args.csv)


def profile_cmd(args: argparse.Namespace) -> None:
    spec = get_workload(args.workload, dtype=args.dtype)
    result = profile_workload(
        spec,
        device=args.device,
        mode=args.mode,
        warmup=args.warmup,
        iterations=args.iterations,
        roofline=_roofline(args),
    )
    _export([result], args)


def compare_cmd(args: argparse.Namespace) -> None:
    spec = get_workload(args.workload, dtype=args.dtype)
    results = [
        profile_workload(
            spec,
            device=args.device,
            mode=mode,
            warmup=args.warmup,
            iterations=args.iterations,
            roofline=_roofline(args),
        )
        for mode in args.modes
    ]
    _export(results, args)


def sweep_cmd(args: argparse.Namespace) -> None:
    base = matmul_estimate(args.m, args.n, args.k, args.dtype)
    rows = []
    for tile in args.tile_sizes:
        tile_stats = tile_utilization(args.m, args.n, args.k, tile)
        rows.append(
            {
                "workload": "matmul_tile_sweep",
                "mode": f"tile={tile}",
                "device": args.device,
                "dtype": args.dtype,
                "time_ms": 0.0,
                "flops": base.flops,
                "memory_bytes": base.memory_bytes,
                "arithmetic_intensity": base.arithmetic_intensity,
                "achieved_gflops": tile_stats["tile_utilization"] * args.peak_tflops * 1000,
                "bottleneck": f"{tile_stats['tile_utilization']:.1%} tile utilization",
                "metadata": tile_stats,
            }
        )
    _export(rows, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile PyTorch tensor workloads with roofline-style estimates.")
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu.")
        subparser.add_argument("--dtype", default="float32", help="Tensor dtype: float16, bfloat16, float32, tf32.")
        subparser.add_argument("--warmup", type=int, default=5)
        subparser.add_argument("--iterations", type=int, default=20)
        subparser.add_argument("--peak-tflops", type=float, default=19.5)
        subparser.add_argument("--bandwidth-gbps", type=float, default=936.0)
        subparser.add_argument("--json", help="Write JSON report to this path.")
        subparser.add_argument("--csv", help="Write CSV report to this path.")

    profile = subparsers.add_parser("profile", help="Profile one workload.")
    add_common(profile)
    profile.add_argument("--workload", choices=["matmul", "conv2d", "attention"], default="matmul")
    profile.add_argument("--mode", choices=["eager", "compile"], default="eager")
    profile.set_defaults(func=profile_cmd)

    compare = subparsers.add_parser("compare", help="Compare eager, torch.compile, and optional Triton.")
    add_common(compare)
    compare.add_argument("--workload", choices=["matmul", "conv2d", "attention"], default="matmul")
    compare.add_argument("--modes", nargs="+", choices=["eager", "compile", "triton"], default=["eager", "compile"])
    compare.set_defaults(func=compare_cmd)

    sweep = subparsers.add_parser("sweep", help="Estimate matrix tile utilization across tile sizes.")
    add_common(sweep)
    sweep.add_argument("--m", type=int, default=4096)
    sweep.add_argument("--n", type=int, default=4096)
    sweep.add_argument("--k", type=int, default=4096)
    sweep.add_argument("--tile-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    sweep.set_defaults(func=sweep_cmd)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
