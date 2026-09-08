from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod
from typing import Callable, Iterable

from .estimator import HardwareRoofline
from .profiler import ProfileResult, profile_workload
from .triton_kernels import TritonMatmulConfig, triton_candidate_configs
from .workloads import get_workload


@dataclass(frozen=True)
class TuningCandidate:
    configuration: str
    mode: str
    implementation: str | None = None
    compile_mode: str | None = None
    triton_config: TritonMatmulConfig | None = None


@dataclass(frozen=True)
class TuningFailure:
    configuration: str
    error: str


@dataclass(frozen=True)
class AutotuneResult:
    workload: str
    baseline_configuration: str
    best_configuration: str
    baseline_time_ms: float
    best_time_ms: float
    latency_reduction_percent: float
    throughput_gain_percent: float
    recommendation: str
    results: list[ProfileResult]
    failures: list[TuningFailure]

    def to_dict(self) -> dict[str, object]:
        return {
            "workload": self.workload,
            "baseline_configuration": self.baseline_configuration,
            "best_configuration": self.best_configuration,
            "baseline_time_ms": self.baseline_time_ms,
            "best_time_ms": self.best_time_ms,
            "latency_reduction_percent": self.latency_reduction_percent,
            "throughput_gain_percent": self.throughput_gain_percent,
            "recommendation": self.recommendation,
            "results": [asdict(result) for result in self.results],
            "failures": [asdict(failure) for failure in self.failures],
        }


ProfileRunner = Callable[..., ProfileResult]


def tuning_candidates(
    workload: str,
    compile_modes: Iterable[str] = ("default", "reduce-overhead", "max-autotune"),
    tile_sizes: Iterable[int] = (32, 64, 128),
    include_triton: bool = True,
) -> list[TuningCandidate]:
    normalized = workload.lower()
    candidates = [TuningCandidate("eager/baseline", "eager")]
    compile_modes = tuple(compile_modes)

    candidates.extend(
        TuningCandidate(f"compile/{compile_mode}", "compile", compile_mode=compile_mode)
        for compile_mode in compile_modes
    )

    if normalized == "attention":
        candidates.append(TuningCandidate("eager/sdpa", "eager", implementation="sdpa"))
        candidates.extend(
            TuningCandidate(
                f"compile/{compile_mode}/sdpa",
                "compile",
                implementation="sdpa",
                compile_mode=compile_mode,
            )
            for compile_mode in compile_modes
        )
    elif normalized == "conv2d":
        candidates.append(
            TuningCandidate("eager/channels-last", "eager", implementation="channels-last")
        )
        candidates.extend(
            TuningCandidate(
                f"compile/{compile_mode}/channels-last",
                "compile",
                implementation="channels-last",
                compile_mode=compile_mode,
            )
            for compile_mode in compile_modes
        )
    elif normalized == "matmul":
        if include_triton:
            candidates.extend(
                TuningCandidate(
                    f"triton/{config.label}",
                    "triton",
                    triton_config=config,
                )
                for config in triton_candidate_configs(tuple(tile_sizes))
            )
    else:
        raise ValueError("Unknown workload. Choose one of: matmul, conv2d, attention")
    return candidates


def _recommendation(baseline: ProfileResult, best: ProfileResult) -> str:
    if best.configuration == baseline.configuration:
        return "Keep the eager baseline; none of the measured candidates was faster."
    if baseline.bottleneck == "memory-bound":
        return (
            f"Use {best.configuration}; the roofline result favors reducing materialization "
            "and memory traffic."
        )
    return (
        f"Use {best.configuration}; the compute-bound result favors compiler and kernel "
        "configuration tuning."
    )


def run_autotune(
    workload: str,
    *,
    device: str = "cuda",
    dtype: str = "float16",
    dimensions: dict[str, int] | None = None,
    warmup: int = 5,
    iterations: int = 20,
    roofline: HardwareRoofline | None = None,
    compile_modes: Iterable[str] = ("default", "reduce-overhead", "max-autotune"),
    tile_sizes: Iterable[int] = (32, 64, 128),
    include_triton: bool = True,
    runner: ProfileRunner = profile_workload,
) -> AutotuneResult:
    dimensions = dimensions or {}
    roofline = roofline or HardwareRoofline()
    results: list[ProfileResult] = []
    failures: list[TuningFailure] = []

    for candidate in tuning_candidates(workload, compile_modes, tile_sizes, include_triton):
        try:
            spec = get_workload(
                workload,
                dtype=dtype,
                implementation=candidate.implementation,
                triton_config=candidate.triton_config,
                **dimensions,
            )
            results.append(
                runner(
                    spec,
                    device=device,
                    mode=candidate.mode,
                    warmup=warmup,
                    iterations=iterations,
                    roofline=roofline,
                    compile_mode=candidate.compile_mode,
                    configuration=candidate.configuration,
                )
            )
        except Exception as exc:  # A failed candidate must not abort the rest of a sweep.
            failures.append(TuningFailure(candidate.configuration, f"{type(exc).__name__}: {exc}"))

    baseline = next((result for result in results if result.configuration == "eager/baseline"), None)
    if baseline is None:
        detail = failures[0].error if failures else "unknown error"
        raise RuntimeError(f"The eager baseline failed: {detail}")

    best = min(results, key=lambda result: result.time_ms)
    latency_reduction = (baseline.time_ms - best.time_ms) / baseline.time_ms * 100
    throughput_gain = (best.achieved_gflops / baseline.achieved_gflops - 1) * 100
    return AutotuneResult(
        workload=workload,
        baseline_configuration=baseline.configuration,
        best_configuration=best.configuration,
        baseline_time_ms=baseline.time_ms,
        best_time_ms=best.time_ms,
        latency_reduction_percent=latency_reduction,
        throughput_gain_percent=throughput_gain,
        recommendation=_recommendation(baseline, best),
        results=results,
        failures=failures,
    )


def diagnosis_payload(tunings: Iterable[AutotuneResult]) -> dict[str, object]:
    tunings = list(tunings)
    if not tunings:
        raise ValueError("At least one autotune result is required.")

    throughput_ratios = [1 + tuning.throughput_gain_percent / 100 for tuning in tunings]
    geometric_mean_gain = (prod(throughput_ratios) ** (1 / len(throughput_ratios)) - 1) * 100
    maximum_gain = max(tuning.throughput_gain_percent for tuning in tunings)
    return {
        "schema_version": 1,
        "workloads": [tuning.to_dict() for tuning in tunings],
        "aggregate": {
            "geometric_mean_throughput_gain_percent": geometric_mean_gain,
            "maximum_workload_throughput_gain_percent": maximum_gain,
        },
    }


def matmul_suite_payload(tunings: Iterable[AutotuneResult]) -> dict[str, object]:
    tunings = list(tunings)
    if not tunings:
        raise ValueError("At least one matmul autotune result is required.")

    summaries = []
    for tuning in tunings:
        if tuning.workload != "matmul":
            raise ValueError("The matmul suite only accepts matmul autotune results.")
        baseline = next(
            result
            for result in tuning.results
            if result.configuration == tuning.baseline_configuration
        )
        shape = {
            dimension: int(baseline.metadata[dimension])
            for dimension in ("m", "n", "k")
        }
        summaries.append(
            {
                "shape": shape,
                "baseline_configuration": tuning.baseline_configuration,
                "best_configuration": tuning.best_configuration,
                "baseline_time_ms": tuning.baseline_time_ms,
                "best_time_ms": tuning.best_time_ms,
                "latency_reduction_percent": tuning.latency_reduction_percent,
                "throughput_gain_percent": tuning.throughput_gain_percent,
                "failed_candidates": len(tuning.failures),
            }
        )

    best_observed = max(summaries, key=lambda summary: summary["throughput_gain_percent"])
    return {
        "schema_version": 1,
        "benchmark": "small-matmul-autotune",
        "results": summaries,
        "best_observed": best_observed,
    }


def render_matmul_suite_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Small Matmul Autotune",
        "",
        "All improvements are measured against eager PyTorch for the same shape and dtype.",
        "",
        "| Shape (M x N x K) | Baseline (ms) | Best configuration | Best (ms) | Latency reduction | Throughput gain |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for result in payload["results"]:
        shape = result["shape"]
        lines.append(
            f"| {shape['m']} x {shape['n']} x {shape['k']} | "
            f"{result['baseline_time_ms']:.4f} | {result['best_configuration']} | "
            f"{result['best_time_ms']:.4f} | "
            f"{result['latency_reduction_percent']:.2f}% | "
            f"{result['throughput_gain_percent']:.2f}% |"
        )

    best = payload["best_observed"]
    shape = best["shape"]
    lines.extend(
        [
            "",
            f"Highest measured throughput gain: {best['throughput_gain_percent']:.2f}% "
            f"at {shape['m']} x {shape['n']} x {shape['k']} using "
            f"{best['best_configuration']}.",
            "",
        ]
    )
    return "\n".join(lines)


def render_diagnosis_markdown(payload: dict[str, object]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# PyTorch GPU Diagnosis",
        "",
        "Measured results are derived from the raw per-configuration timings in the JSON report.",
        "",
        "| Workload | Baseline (ms) | Best configuration | Best (ms) | Throughput gain |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for workload in payload["workloads"]:
        lines.append(
            f"| {workload['workload']} | {workload['baseline_time_ms']:.3f} | "
            f"{workload['best_configuration']} | {workload['best_time_ms']:.3f} | "
            f"{workload['throughput_gain_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Geometric-mean throughput gain: "
            f"{aggregate['geometric_mean_throughput_gain_percent']:.2f}%.",
            "",
        ]
    )
    return "\n".join(lines)
