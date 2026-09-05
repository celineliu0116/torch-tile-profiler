from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TritonMatmulConfig:
    block_m: int = 32
    block_n: int = 32
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 2

    def __post_init__(self) -> None:
        for name in ("block_m", "block_n", "block_k", "num_warps", "num_stages"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")

    @property
    def label(self) -> str:
        return (
            f"bm{self.block_m}-bn{self.block_n}-bk{self.block_k}"
            f"-w{self.num_warps}-s{self.num_stages}"
        )


def triton_candidate_configs(tile_sizes: list[int] | tuple[int, ...] = (32, 64, 128)) -> list[TritonMatmulConfig]:
    configs = []
    for tile in tile_sizes:
        warps = 4 if tile <= 64 else 8
        stages = 2 if tile <= 32 else 3
        configs.append(TritonMatmulConfig(tile, tile, 32, warps, stages))
    if 64 in tile_sizes and 128 in tile_sizes:
        configs.extend(
            [
                TritonMatmulConfig(64, 128, 32, 4, 3),
                TritonMatmulConfig(128, 64, 32, 4, 3),
            ]
        )
    return configs


def make_triton_matmul(
    torch: object,
    a: object,
    b: object,
    config: TritonMatmulConfig | None = None,
) -> Callable[[], object]:
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Triton mode requires Triton. Install it with: pip install -e '.[triton]'") from exc

    config = config or TritonMatmulConfig()

    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        acc = tl.zeros((block_m, block_n), tl.float32)

        for k_start in range(0, k, block_k):
            k_idxs = k_start + offs_k
            a_vals = tl.load(a_ptr + offs_m[:, None] * k + k_idxs[None, :], mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k), other=0.0)
            b_vals = tl.load(b_ptr + k_idxs[:, None] * n + offs_n[None, :], mask=(k_idxs[:, None] < k) & (offs_n[None, :] < n), other=0.0)
            acc += tl.dot(a_vals, b_vals)

        tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    def run() -> object:
        grid = (triton.cdiv(m, config.block_m), triton.cdiv(n, config.block_n))
        _matmul_kernel[grid](
            a,
            b,
            c,
            m,
            n,
            k,
            config.block_m,
            config.block_n,
            config.block_k,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        return c

    return run
