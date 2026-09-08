from dataclasses import replace

import pytest

from torch_tile_profiler.autotune import (
    AutotuneResult,
    diagnosis_payload,
    matmul_suite_payload,
    render_matmul_suite_markdown,
    run_autotune,
    tuning_candidates,
)
from torch_tile_profiler.profiler import ProfileResult
from torch_tile_profiler.triton_kernels import TritonMatmulConfig, triton_candidate_configs


def _profile(
    configuration: str,
    time_ms: float,
    *,
    workload: str = "matmul",
    bottleneck: str = "compute-bound",
    accelerator: str = "NVIDIA T4",
) -> ProfileResult:
    flops = 2 * 4096**3
    return ProfileResult(
        workload=workload,
        mode=configuration.split("/", 1)[0],
        configuration=configuration,
        device="cuda",
        dtype="float16",
        time_ms=time_ms,
        flops=flops,
        memory_bytes=1,
        arithmetic_intensity=float(flops),
        achieved_gflops=flops / (time_ms / 1000) / 1e9,
        bottleneck=bottleneck,
        profiler_key_averages=[],
        metadata={"m": 4096, "n": 4096, "k": 4096, "accelerator": accelerator},
    )


def test_triton_candidate_configs_include_rectangular_tiles() -> None:
    configs = triton_candidate_configs((32, 64, 128))
    assert TritonMatmulConfig(64, 128, 32, 4, 3) in configs
    assert len({config.label for config in configs}) == len(configs)


def test_candidates_follow_roofline_targeted_optimizations() -> None:
    attention = tuning_candidates("attention", compile_modes=("default",), include_triton=False)
    conv = tuning_candidates("conv2d", compile_modes=("default",), include_triton=False)
    assert any(candidate.implementation == "sdpa" for candidate in attention)
    assert any(candidate.implementation == "channels-last" for candidate in conv)


def test_autotune_ranks_measured_results_and_keeps_failures() -> None:
    times = {"eager/baseline": 10.0, "compile/default": 8.0}

    def fake_runner(spec, **kwargs):
        configuration = kwargs["configuration"]
        if configuration == "compile/max-autotune":
            raise RuntimeError("unsupported")
        return _profile(configuration, times[configuration])

    tuning = run_autotune(
        "matmul",
        compile_modes=("default", "max-autotune"),
        include_triton=False,
        runner=fake_runner,
    )
    assert tuning.best_configuration == "compile/default"
    assert tuning.latency_reduction_percent == pytest.approx(20.0)
    assert tuning.throughput_gain_percent == pytest.approx(25.0)
    assert tuning.failures[0].configuration == "compile/max-autotune"


def test_diagnosis_reports_aggregate_metrics() -> None:
    matmul_baseline = _profile("eager/baseline", 10.0)
    matmul_best = _profile("compile/max-autotune", 8.5)
    matmul = AutotuneResult(
        "matmul",
        "eager/baseline",
        "compile/max-autotune",
        10.0,
        8.5,
        15.0,
        matmul_best.achieved_gflops / matmul_baseline.achieved_gflops * 100 - 100,
        "use compiled",
        [matmul_baseline, matmul_best],
        [],
    )
    conv = replace(
        matmul,
        workload="conv2d",
        results=[_profile("eager/baseline", 10.0, workload="conv2d"), _profile("compile/default", 8.0, workload="conv2d")],
        throughput_gain_percent=25.0,
    )
    attention = replace(
        matmul,
        workload="attention",
        results=[_profile("eager/baseline", 10.0, workload="attention", bottleneck="memory-bound"), _profile("eager/sdpa", 7.0, workload="attention", bottleneck="compute-bound")],
        throughput_gain_percent=40.0,
    )
    payload = diagnosis_payload([matmul, conv, attention])
    aggregate = payload["aggregate"]
    assert set(payload) == {"schema_version", "workloads", "aggregate"}
    assert aggregate["maximum_workload_throughput_gain_percent"] == pytest.approx(40.0)
    assert aggregate["geometric_mean_throughput_gain_percent"] > 0


def test_small_matmul_suite_reports_highest_measured_gain() -> None:
    baseline_512 = replace(
        _profile("eager/baseline", 1.0),
        metadata={"m": 512, "n": 512, "k": 512, "accelerator": "NVIDIA T4"},
    )
    best_512 = replace(
        _profile("triton/bm64", 0.8),
        metadata={"m": 512, "n": 512, "k": 512, "accelerator": "NVIDIA T4"},
    )
    tuning_512 = AutotuneResult(
        "matmul",
        "eager/baseline",
        "triton/bm64",
        1.0,
        0.8,
        20.0,
        25.0,
        "use triton",
        [baseline_512, best_512],
        [],
    )
    baseline_1024 = replace(
        _profile("eager/baseline", 2.0),
        metadata={"m": 1024, "n": 1024, "k": 1024, "accelerator": "NVIDIA T4"},
    )
    best_1024 = replace(
        _profile("compile/default", 1.8),
        metadata={"m": 1024, "n": 1024, "k": 1024, "accelerator": "NVIDIA T4"},
    )
    tuning_1024 = replace(
        tuning_512,
        best_configuration="compile/default",
        baseline_time_ms=2.0,
        best_time_ms=1.8,
        latency_reduction_percent=10.0,
        throughput_gain_percent=11.111111,
        results=[baseline_1024, best_1024],
    )

    payload = matmul_suite_payload([tuning_1024, tuning_512])
    assert payload["best_observed"]["shape"] == {"m": 512, "n": 512, "k": 512}
    assert payload["best_observed"]["throughput_gain_percent"] == pytest.approx(25.0)
    markdown = render_matmul_suite_markdown(payload)
    assert "512 x 512 x 512" in markdown
    assert "Highest measured throughput gain: 25.00%" in markdown
