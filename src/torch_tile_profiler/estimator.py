from __future__ import annotations

from dataclasses import dataclass
from math import prod


DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "tf32": 4,
    "float64": 8,
    "int8": 1,
}


@dataclass(frozen=True)
class HardwareRoofline:
    peak_tflops: float = 19.5
    bandwidth_gbps: float = 936.0

    @property
    def ridge_point(self) -> float:
        """FLOPs per byte needed to become compute-bound."""
        return (self.peak_tflops * 1e12) / (self.bandwidth_gbps * 1e9)


@dataclass(frozen=True)
class OperationEstimate:
    workload: str
    flops: int
    memory_bytes: int
    dtype: str
    metadata: dict[str, int | float | str]

    @property
    def arithmetic_intensity(self) -> float:
        if self.memory_bytes == 0:
            return float("inf")
        return self.flops / self.memory_bytes


def dtype_size(dtype: str) -> int:
    normalized = dtype.replace("torch.", "")
    if normalized not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype '{dtype}'. Known dtypes: {', '.join(DTYPE_BYTES)}")
    return DTYPE_BYTES[normalized]


def classify_bottleneck(ai_flops_per_byte: float, roofline: HardwareRoofline) -> str:
    return "memory-bound" if ai_flops_per_byte < roofline.ridge_point else "compute-bound"


def matmul_estimate(m: int, n: int, k: int, dtype: str = "float32") -> OperationEstimate:
    itemsize = dtype_size(dtype)
    flops = 2 * m * n * k
    memory_bytes = (m * k + k * n + m * n) * itemsize
    return OperationEstimate("matmul", flops, memory_bytes, dtype, {"m": m, "n": n, "k": k})


def conv2d_estimate(
    batch: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel: int,
    dtype: str = "float32",
    stride: int = 1,
    padding: int = 1,
) -> OperationEstimate:
    itemsize = dtype_size(dtype)
    out_h = ((height + 2 * padding - kernel) // stride) + 1
    out_w = ((width + 2 * padding - kernel) // stride) + 1
    output_elems = batch * out_channels * out_h * out_w
    flops = 2 * output_elems * in_channels * kernel * kernel
    input_bytes = batch * in_channels * height * width * itemsize
    weight_bytes = out_channels * in_channels * kernel * kernel * itemsize
    output_bytes = output_elems * itemsize
    return OperationEstimate(
        "conv2d",
        flops,
        input_bytes + weight_bytes + output_bytes,
        dtype,
        {
            "batch": batch,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "height": height,
            "width": width,
            "kernel": kernel,
            "stride": stride,
            "padding": padding,
            "out_h": out_h,
            "out_w": out_w,
        },
    )


def attention_estimate(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    dtype: str = "float32",
) -> OperationEstimate:
    itemsize = dtype_size(dtype)
    qkv_elems = 3 * batch * heads * seq_len * head_dim
    scores_elems = batch * heads * seq_len * seq_len
    output_elems = batch * heads * seq_len * head_dim
    qk_flops = 2 * batch * heads * seq_len * seq_len * head_dim
    av_flops = qk_flops
    scale_flops = scores_elems
    softmax_flops = 5 * scores_elems
    flops = qk_flops + scale_flops + softmax_flops + av_flops
    memory_bytes = (qkv_elems + scores_elems + output_elems) * itemsize
    return OperationEstimate(
        "attention",
        flops,
        memory_bytes,
        dtype,
        {
            "batch": batch,
            "heads": heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "qk_flops": qk_flops,
            "scale_flops": scale_flops,
            "softmax_flops": softmax_flops,
            "av_flops": av_flops,
        },
    )


def tile_utilization(m: int, n: int, k: int, tile: int) -> dict[str, int | float]:
    tiles_m = (m + tile - 1) // tile
    tiles_n = (n + tile - 1) // tile
    tiles_k = (k + tile - 1) // tile
    padded = tiles_m * tile * tiles_n * tile * tiles_k * tile
    useful = m * n * k
    utilization = useful / padded if padded else 0.0
    return {
        "tile": tile,
        "tiles_m": tiles_m,
        "tiles_n": tiles_n,
        "tiles_k": tiles_k,
        "useful_elements": useful,
        "padded_elements": padded,
        "tile_utilization": utilization,
    }


def tensor_bytes(shape: tuple[int, ...], dtype: str) -> int:
    return prod(shape) * dtype_size(dtype)
