from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .estimator import OperationEstimate, attention_estimate, conv2d_estimate, matmul_estimate


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    estimate: OperationEstimate
    make_callable: Callable[[object, str, str], Callable[[], object]]


def _torch_dtype(torch: object, dtype: str) -> object:
    try:
        return getattr(torch, dtype.replace("torch.", ""))
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype}") from exc


def matmul_workload(
    m: int = 4096,
    n: int = 4096,
    k: int = 4096,
    dtype: str = "float32",
) -> WorkloadSpec:
    def make(torch: object, device: str, mode: str) -> Callable[[], object]:
        torch_dtype = _torch_dtype(torch, dtype)
        a = torch.randn((m, k), device=device, dtype=torch_dtype)
        b = torch.randn((k, n), device=device, dtype=torch_dtype)

        if mode == "triton":
            if not device.startswith("cuda"):
                raise RuntimeError("Triton mode requires a CUDA device.")
            from .triton_kernels import make_triton_matmul

            return make_triton_matmul(torch, a, b)

        def run() -> object:
            return torch.matmul(a, b)

        return run

    return WorkloadSpec("matmul", matmul_estimate(m, n, k, dtype), make)


def conv2d_workload(
    batch: int = 32,
    in_channels: int = 64,
    out_channels: int = 128,
    height: int = 56,
    width: int = 56,
    kernel: int = 3,
    dtype: str = "float32",
) -> WorkloadSpec:
    def make(torch: object, device: str, mode: str) -> Callable[[], object]:
        torch_dtype = _torch_dtype(torch, dtype)
        x = torch.randn((batch, in_channels, height, width), device=device, dtype=torch_dtype)
        weight = torch.randn((out_channels, in_channels, kernel, kernel), device=device, dtype=torch_dtype)

        def run() -> object:
            return torch.nn.functional.conv2d(x, weight, padding=kernel // 2)

        return run

    estimate = conv2d_estimate(batch, in_channels, out_channels, height, width, kernel, dtype)
    return WorkloadSpec("conv2d", estimate, make)


def attention_workload(
    batch: int = 8,
    heads: int = 16,
    seq_len: int = 1024,
    head_dim: int = 64,
    dtype: str = "float32",
) -> WorkloadSpec:
    def make(torch: object, device: str, mode: str) -> Callable[[], object]:
        torch_dtype = _torch_dtype(torch, dtype)
        q = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch_dtype)
        key = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch_dtype)
        value = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch_dtype)
        scale = head_dim**-0.5

        def run() -> object:
            scores = torch.matmul(q, key.transpose(-2, -1)) * scale
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, value)

        return run

    return WorkloadSpec("attention", attention_estimate(batch, heads, seq_len, head_dim, dtype), make)


def get_workload(name: str, dtype: str = "float32") -> WorkloadSpec:
    normalized = name.lower()
    if normalized == "matmul":
        return matmul_workload(dtype=dtype)
    if normalized == "conv2d":
        return conv2d_workload(dtype=dtype)
    if normalized in {"attention", "transformer"}:
        return attention_workload(dtype=dtype)
    raise ValueError("Unknown workload. Choose one of: matmul, conv2d, attention")
