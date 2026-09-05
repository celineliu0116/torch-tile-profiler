from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from .estimator import HardwareRoofline, classify_bottleneck
from .workloads import WorkloadSpec


@dataclass(frozen=True)
class ProfileResult:
    workload: str
    mode: str
    configuration: str
    device: str
    dtype: str
    time_ms: float
    flops: int
    memory_bytes: int
    arithmetic_intensity: float
    achieved_gflops: float
    bottleneck: str
    profiler_key_averages: list[dict[str, float | str]]
    metadata: dict[str, int | float | str]


def _import_torch() -> object:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to run profiles. Install it with: pip install -e '.[torch]'"
        ) from exc
    return torch


def _sync(torch: object, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("mps") and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _maybe_compile(
    torch: object,
    fn: Callable[[], object],
    mode: str,
    compile_mode: str | None,
) -> Callable[[], object]:
    if mode in {"eager", "triton"}:
        return fn
    if mode == "compile":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build.")
        return torch.compile(fn, mode=compile_mode) if compile_mode else torch.compile(fn)
    raise ValueError("Mode must be 'eager', 'compile', or 'triton'.")


def profile_workload(
    spec: WorkloadSpec,
    device: str = "cuda",
    mode: str = "eager",
    warmup: int = 5,
    iterations: int = 20,
    roofline: HardwareRoofline | None = None,
    compile_mode: str | None = None,
    configuration: str | None = None,
) -> ProfileResult:
    torch = _import_torch()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    if warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    roofline = roofline or HardwareRoofline()
    run = _maybe_compile(torch, spec.make_callable(torch, device, mode), mode, compile_mode)

    for _ in range(warmup):
        run()
    _sync(torch, device)

    if device.startswith("cuda"):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            run()
        end_event.record()
        _sync(torch, device)
        elapsed_ms = float(start_event.elapsed_time(end_event)) / iterations
    else:
        start = perf_counter()
        for _ in range(iterations):
            run()
        _sync(torch, device)
        elapsed_ms = (perf_counter() - start) * 1000 / iterations

    # Capture a representative trace separately so profiler overhead cannot skew
    # the benchmark timing used for ranking configurations.
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        run()
        _sync(torch, device)

    achieved_gflops = spec.estimate.flops / (elapsed_ms / 1000) / 1e9
    bottleneck = classify_bottleneck(spec.estimate.arithmetic_intensity, roofline)
    key_averages = []
    for event in prof.key_averages()[:12]:
        key_averages.append(
            {
                "name": event.key,
                "cpu_time_total_us": float(event.cpu_time_total),
                "cuda_time_total_us": float(getattr(event, "cuda_time_total", 0.0)),
            }
        )

    metadata = dict(spec.estimate.metadata)
    metadata["implementation"] = spec.implementation
    metadata["correctness_verification"] = spec.verification
    metadata["torch_version"] = str(torch.__version__)
    metadata["warmup_iterations"] = warmup
    metadata["measured_iterations"] = iterations
    metadata["timing_method"] = "cuda-events" if device.startswith("cuda") else "perf-counter"
    if compile_mode:
        metadata["compile_mode"] = compile_mode
    if device.startswith("cuda"):
        metadata["accelerator"] = torch.cuda.get_device_name(torch.cuda.current_device())

    return ProfileResult(
        workload=spec.name,
        mode=mode,
        configuration=configuration or mode,
        device=device,
        dtype=spec.estimate.dtype,
        time_ms=elapsed_ms,
        flops=spec.estimate.flops,
        memory_bytes=spec.estimate.memory_bytes,
        arithmetic_intensity=spec.estimate.arithmetic_intensity,
        achieved_gflops=achieved_gflops,
        bottleneck=bottleneck,
        profiler_key_averages=key_averages,
        metadata=metadata,
    )
