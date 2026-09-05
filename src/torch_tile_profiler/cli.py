from __future__ import annotations

import argparse
from pathlib import Path

from .autotune import diagnosis_payload, render_diagnosis_markdown, run_autotune
from .estimator import HardwareRoofline, matmul_estimate, tile_utilization
from .profiler import profile_workload
from .reports import print_table, write_csv, write_json, write_json_payload
from .triton_kernels import TritonMatmulConfig
from .workloads import get_workload


WORKLOAD_DEFAULTS = {
    "matmul": {"m": 4096, "n": 4096, "k": 4096},
    "conv2d": {
        "batch": 32,
        "in_channels": 64,
        "out_channels": 128,
        "height": 56,
        "width": 56,
        "kernel": 3,
    },
    "attention": {"batch": 8, "heads": 16, "seq_len": 1024, "head_dim": 64},
}


def _roofline(args: argparse.Namespace) -> HardwareRoofline:
    return HardwareRoofline(peak_tflops=args.peak_tflops, bandwidth_gbps=args.bandwidth_gbps)


def _dimensions(args: argparse.Namespace, workload: str) -> dict[str, int]:
    return {
        name: getattr(args, name, None) or default
        for name, default in WORKLOAD_DEFAULTS[workload].items()
    }


def _export(results: list[object], args: argparse.Namespace) -> None:
    print_table(results)
    if getattr(args, "json", None):
        write_json(results, args.json)
    if getattr(args, "csv", None):
        write_csv(results, args.csv)


def _triton_config(args: argparse.Namespace) -> TritonMatmulConfig:
    return TritonMatmulConfig(
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def profile_cmd(args: argparse.Namespace) -> None:
    triton_config = _triton_config(args) if args.mode == "triton" else None
    spec = get_workload(
        args.workload,
        dtype=args.dtype,
        implementation=args.implementation,
        triton_config=triton_config,
        **_dimensions(args, args.workload),
    )
    configuration = args.mode
    if args.mode == "compile":
        configuration = f"compile/{args.compile_mode}"
    elif args.mode == "triton":
        configuration = f"triton/{triton_config.label}"
    result = profile_workload(
        spec,
        device=args.device,
        mode=args.mode,
        warmup=args.warmup,
        iterations=args.iterations,
        roofline=_roofline(args),
        compile_mode=args.compile_mode if args.mode == "compile" else None,
        configuration=configuration,
    )
    _export([result], args)


def compare_cmd(args: argparse.Namespace) -> None:
    results = []
    for mode in args.modes:
        triton_config = _triton_config(args) if mode == "triton" else None
        spec = get_workload(
            args.workload,
            dtype=args.dtype,
            implementation=args.implementation,
            triton_config=triton_config,
            **_dimensions(args, args.workload),
        )
        configuration = mode
        if mode == "compile":
            configuration = f"compile/{args.compile_mode}"
        elif mode == "triton":
            configuration = f"triton/{triton_config.label}"
        results.append(
            profile_workload(
                spec,
                device=args.device,
                mode=mode,
                warmup=args.warmup,
                iterations=args.iterations,
                roofline=_roofline(args),
                compile_mode=args.compile_mode if mode == "compile" else None,
                configuration=configuration,
            )
        )
    _export(results, args)


def sweep_cmd(args: argparse.Namespace) -> None:
    if args.measure:
        tuning = run_autotune(
            "matmul",
            device=args.device,
            dtype=args.dtype,
            dimensions={"m": args.m, "n": args.n, "k": args.k},
            warmup=args.warmup,
            iterations=args.iterations,
            roofline=_roofline(args),
            compile_modes=(),
            tile_sizes=args.tile_sizes,
            include_triton=True,
        )
        _export(tuning.results, args)
        return

    base = matmul_estimate(args.m, args.n, args.k, args.dtype)
    rows = []
    for tile in args.tile_sizes:
        tile_stats = tile_utilization(args.m, args.n, args.k, tile)
        rows.append(
            {
                "workload": "matmul_tile_sweep",
                "mode": "analytical",
                "configuration": f"tile={tile}",
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


def autotune_cmd(args: argparse.Namespace) -> None:
    include_triton = args.include_triton
    if include_triton is None:
        include_triton = args.device.startswith("cuda")
    tuning = run_autotune(
        args.workload,
        device=args.device,
        dtype=args.dtype,
        dimensions=_dimensions(args, args.workload),
        warmup=args.warmup,
        iterations=args.iterations,
        roofline=_roofline(args),
        compile_modes=args.compile_modes,
        tile_sizes=args.tile_sizes,
        include_triton=include_triton,
    )
    print_table(tuning.results)
    print(tuning.recommendation)
    print(f"Measured throughput gain: {tuning.throughput_gain_percent:.2f}%")
    if tuning.failures:
        print(f"Skipped {len(tuning.failures)} failed candidate(s); see the JSON report for details.")
    if args.json:
        write_json_payload(tuning.to_dict(), args.json)
    if args.csv:
        write_csv(tuning.results, args.csv)


def diagnose_cmd(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_triton = args.include_triton
    if include_triton is None:
        include_triton = args.device.startswith("cuda")

    tunings = []
    for workload in args.workloads:
        tuning = run_autotune(
            workload,
            device=args.device,
            dtype=args.dtype,
            dimensions=_dimensions(args, workload),
            warmup=args.warmup,
            iterations=args.iterations,
            roofline=_roofline(args),
            compile_modes=args.compile_modes,
            tile_sizes=args.tile_sizes,
            include_triton=include_triton and workload == "matmul",
        )
        tunings.append(tuning)
        print_table(tuning.results)
        write_json_payload(tuning.to_dict(), output_dir / f"{workload}.json")
        write_csv(tuning.results, output_dir / f"{workload}.csv")

    payload = diagnosis_payload(tunings)
    write_json_payload(payload, output_dir / "diagnosis.json")
    (output_dir / "diagnosis.md").write_text(
        render_diagnosis_markdown(payload), encoding="utf-8"
    )
    print(f"Wrote diagnosis reports to {output_dir}")


def _add_common(subparser: argparse.ArgumentParser, *, dtype: str = "float32") -> None:
    subparser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu.")
    subparser.add_argument("--dtype", default=dtype, help="Tensor dtype: float16, bfloat16, float32.")
    subparser.add_argument("--warmup", type=int, default=5)
    subparser.add_argument("--iterations", type=int, default=20)
    subparser.add_argument("--peak-tflops", type=float, default=19.5)
    subparser.add_argument("--bandwidth-gbps", type=float, default=936.0)


def _add_dimensions(subparser: argparse.ArgumentParser) -> None:
    for name in ("m", "n", "k", "batch", "in_channels", "out_channels", "height", "width", "kernel", "heads", "seq_len", "head_dim"):
        subparser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=int)


def _add_exports(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--json", help="Write JSON report to this path.")
    subparser.add_argument("--csv", help="Write CSV report to this path.")


def _add_triton_config(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--block-m", type=int, default=32)
    subparser.add_argument("--block-n", type=int, default=32)
    subparser.add_argument("--block-k", type=int, default=32)
    subparser.add_argument("--num-warps", type=int, default=4)
    subparser.add_argument("--num-stages", type=int, default=2)


def _add_tuning(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--compile-modes",
        nargs="+",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=["default", "reduce-overhead", "max-autotune"],
    )
    subparser.add_argument("--tile-sizes", nargs="+", type=int, default=[32, 64, 128])
    subparser.add_argument(
        "--include-triton",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable measured Triton configurations (defaults on for CUDA).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile and autotune PyTorch tensor workloads with roofline analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Profile one workload and configuration.")
    _add_common(profile)
    _add_dimensions(profile)
    _add_exports(profile)
    _add_triton_config(profile)
    profile.add_argument("--workload", choices=list(WORKLOAD_DEFAULTS), default="matmul")
    profile.add_argument("--mode", choices=["eager", "compile", "triton"], default="eager")
    profile.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    profile.add_argument("--implementation", help="manual/sdpa for attention or contiguous/channels-last for conv2d.")
    profile.set_defaults(func=profile_cmd)

    compare = subparsers.add_parser("compare", help="Compare eager, compiled PyTorch, and Triton.")
    _add_common(compare)
    _add_dimensions(compare)
    _add_exports(compare)
    _add_triton_config(compare)
    compare.add_argument("--workload", choices=list(WORKLOAD_DEFAULTS), default="matmul")
    compare.add_argument("--modes", nargs="+", choices=["eager", "compile", "triton"], default=["eager", "compile"])
    compare.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    compare.add_argument("--implementation")
    compare.set_defaults(func=compare_cmd)

    sweep = subparsers.add_parser("sweep", help="Analyze or benchmark matrix tile sizes.")
    _add_common(sweep)
    _add_exports(sweep)
    sweep.add_argument("--m", type=int, default=4096)
    sweep.add_argument("--n", type=int, default=4096)
    sweep.add_argument("--k", type=int, default=4096)
    sweep.add_argument("--tile-sizes", nargs="+", type=int, default=[32, 64, 128])
    sweep.add_argument("--measure", action="store_true", help="Benchmark each Triton tile configuration on CUDA.")
    sweep.set_defaults(func=sweep_cmd)

    autotune = subparsers.add_parser("autotune", help="Benchmark candidates and select the fastest measured configuration.")
    _add_common(autotune, dtype="float16")
    _add_dimensions(autotune)
    _add_exports(autotune)
    _add_tuning(autotune)
    autotune.add_argument("--workload", choices=list(WORKLOAD_DEFAULTS), default="matmul")
    autotune.set_defaults(func=autotune_cmd)

    diagnose = subparsers.add_parser("diagnose", help="Run the complete profiling, roofline, and autotuning workflow.")
    _add_common(diagnose, dtype="float16")
    _add_dimensions(diagnose)
    _add_tuning(diagnose)
    diagnose.add_argument("--workloads", nargs="+", choices=list(WORKLOAD_DEFAULTS), default=list(WORKLOAD_DEFAULTS))
    diagnose.add_argument("--output-dir", default="reports/latest")
    diagnose.set_defaults(func=diagnose_cmd)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
