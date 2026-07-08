from torch_tile_profiler.estimator import (
    HardwareRoofline,
    attention_estimate,
    classify_bottleneck,
    conv2d_estimate,
    matmul_estimate,
    tile_utilization,
)


def test_matmul_estimate_flops_and_memory() -> None:
    estimate = matmul_estimate(2, 3, 4, "float32")
    assert estimate.flops == 48
    assert estimate.memory_bytes == (2 * 4 + 4 * 3 + 2 * 3) * 4
    assert estimate.arithmetic_intensity == 48 / 104


def test_conv2d_estimate_uses_output_shape() -> None:
    estimate = conv2d_estimate(1, 3, 16, 32, 32, 3, "float32", padding=1)
    assert estimate.metadata["out_h"] == 32
    assert estimate.metadata["out_w"] == 32
    assert estimate.flops == 2 * 1 * 16 * 32 * 32 * 3 * 3 * 3


def test_attention_estimate_contains_softmax_work() -> None:
    estimate = attention_estimate(batch=2, heads=4, seq_len=128, head_dim=64, dtype="float16")
    matmul_flops = 2 * 2 * 4 * 128 * 128 * 64 * 2
    scale_flops = 2 * 4 * 128 * 128
    softmax_flops = 5 * 2 * 4 * 128 * 128
    assert estimate.flops == matmul_flops + scale_flops + softmax_flops
    assert estimate.metadata["scale_flops"] == scale_flops


def test_roofline_classification() -> None:
    roofline = HardwareRoofline(peak_tflops=10, bandwidth_gbps=1000)
    assert roofline.ridge_point == 10
    assert classify_bottleneck(5, roofline) == "memory-bound"
    assert classify_bottleneck(20, roofline) == "compute-bound"


def test_tile_utilization_accounts_for_padding() -> None:
    stats = tile_utilization(130, 128, 64, 64)
    assert stats["tiles_m"] == 3
    assert stats["tiles_n"] == 2
    assert stats["tiles_k"] == 1
    assert 0 < stats["tile_utilization"] < 1
