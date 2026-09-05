from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .estimator import OperationEstimate, attention_estimate, conv2d_estimate, matmul_estimate

if TYPE_CHECKING:
    from .triton_kernels import TritonMatmulConfig


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    estimate: OperationEstimate
    make_callable: Callable[[object, str, str], Callable[[], object]]
    implementation: str = "baseline"
    verification: str = "trusted PyTorch operator"


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
    triton_config: TritonMatmulConfig | None = None,
) -> WorkloadSpec:
    def make(torch: object, device: str, mode: str) -> Callable[[], object]:
        torch_dtype = _torch_dtype(torch, dtype)
        a = torch.randn((m, k), device=device, dtype=torch_dtype)
        b = torch.randn((k, n), device=device, dtype=torch_dtype)

        if mode == "triton":
            if not device.startswith("cuda"):
                raise RuntimeError("Triton mode requires a CUDA device.")
            from .triton_kernels import TritonMatmulConfig, make_triton_matmul

            run = make_triton_matmul(torch, a, b, triton_config or TritonMatmulConfig())
            actual = run()
            expected = torch.matmul(a, b)
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
            return run

        def run() -> object:
            return torch.matmul(a, b)

        return run

    implementation = triton_config.label if triton_config else "torch.matmul"
    verification = "torch.testing.assert_close against torch.matmul" if triton_config else "trusted PyTorch operator"
    return WorkloadSpec(
        "matmul",
        matmul_estimate(m, n, k, dtype),
        make,
        implementation,
        verification,
    )


def conv2d_workload(
    batch: int = 32,
    in_channels: int = 64,
    out_channels: int = 128,
    height: int = 56,
    width: int = 56,
    kernel: int = 3,
    dtype: str = "float32",
    implementation: str = "contiguous",
) -> WorkloadSpec:
    if implementation not in {"contiguous", "channels-last"}:
        raise ValueError("conv2d implementation must be 'contiguous' or 'channels-last'.")

    def make(torch: object, device: str, mode: str) -> Callable[[], object]:
        torch_dtype = _torch_dtype(torch, dtype)
        x = torch.randn((batch, in_channels, height, width), device=device, dtype=torch_dtype)
        weight = torch.randn((out_channels, in_channels, kernel, kernel), device=device, dtype=torch_dtype)
        if implementation == "channels-last":
            x = x.contiguous(memory_format=torch.channels_last)
            weight = weight.contiguous(memory_format=torch.channels_last)

        def run() -> object:
            return torch.nn.functional.conv2d(x, weight, padding=kernel // 2)

        return run

    estimate = conv2d_estimate(batch, in_channels, out_channels, height, width, kernel, dtype)
    return WorkloadSpec("conv2d", estimate, make, implementation)


def attention_workload(
    batch: int = 8,
    heads: int = 16,
    seq_len: int = 1024,
    head_dim: int = 64,
    dtype: str = "float32",
    implementation: str = "manual",
) -> WorkloadSpec:
    if implementation not in {"manual", "sdpa"}:
        raise ValueError("attention implementation must be 'manual' or 'sdpa'.")

    def make(torch: object, device: str, mode: str) -> Callable[[], object]:
        torch_dtype = _torch_dtype(torch, dtype)
        q = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch_dtype)
        key = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch_dtype)
        value = torch.randn((batch, heads, seq_len, head_dim), device=device, dtype=torch_dtype)
        scale = head_dim**-0.5

        if implementation == "sdpa":
            if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                raise RuntimeError("This PyTorch build does not provide scaled_dot_product_attention.")

            def run() -> object:
                return torch.nn.functional.scaled_dot_product_attention(q, key, value)
        else:
            def run() -> object:
                scores = torch.matmul(q, key.transpose(-2, -1)) * scale
                probs = torch.softmax(scores, dim=-1)
                return torch.matmul(probs, value)

        return run

    return WorkloadSpec(
        "attention",
        attention_estimate(batch, heads, seq_len, head_dim, dtype, fused=implementation == "sdpa"),
        make,
        implementation,
    )


def get_workload(
    name: str,
    dtype: str = "float32",
    implementation: str | None = None,
    triton_config: TritonMatmulConfig | None = None,
    **dimensions: int,
) -> WorkloadSpec:
    normalized = name.lower()
    if normalized == "matmul":
        return matmul_workload(dtype=dtype, triton_config=triton_config, **dimensions)
    if normalized == "conv2d":
        return conv2d_workload(dtype=dtype, implementation=implementation or "contiguous", **dimensions)
    if normalized in {"attention", "transformer"}:
        return attention_workload(dtype=dtype, implementation=implementation or "manual", **dimensions)
    raise ValueError("Unknown workload. Choose one of: matmul, conv2d, attention")
