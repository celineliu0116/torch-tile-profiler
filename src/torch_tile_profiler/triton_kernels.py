from __future__ import annotations

from typing import Callable


def make_triton_matmul(torch: object, a: object, b: object, block_size: int = 32) -> Callable[[], object]:
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Triton mode requires Triton. Install it with: pip install -e '.[triton]'") from exc

    @triton.jit
    def _matmul_kernel(a_ptr, b_ptr, c_ptr, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr, block: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * block + tl.arange(0, block)
        offs_n = pid_n * block + tl.arange(0, block)
        offs_k = tl.arange(0, block)
        acc = tl.zeros((block, block), tl.float32)

        for k_start in range(0, k, block):
            k_idxs = k_start + offs_k
            a_vals = tl.load(a_ptr + offs_m[:, None] * k + k_idxs[None, :], mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k), other=0.0)
            b_vals = tl.load(b_ptr + k_idxs[:, None] * n + offs_n[None, :], mask=(k_idxs[:, None] < k) & (offs_n[None, :] < n), other=0.0)
            acc += tl.dot(a_vals, b_vals)

        tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]

    def run() -> object:
        c = torch.empty((m, n), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(m, block_size), triton.cdiv(n, block_size))
        _matmul_kernel[grid](a, b, c, m, n, k, block_size)
        return c

    return run
