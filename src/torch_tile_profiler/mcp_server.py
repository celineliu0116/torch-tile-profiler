from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .autotune import diagnosis_payload, render_diagnosis_markdown, run_autotune
from .estimator import HardwareRoofline, matmul_estimate, tile_utilization
from .profiler import profile_workload
from .reports import write_csv, write_json_payload
from .workloads import get_workload


SERVER_INFO = {"name": "torch-tile-profiler", "version": "0.2.0"}
DIMENSION_NAMES = {
    "m",
    "n",
    "k",
    "batch",
    "in_channels",
    "out_channels",
    "height",
    "width",
    "kernel",
    "heads",
    "seq_len",
    "head_dim",
}


TOOLS = [
    {
        "name": "profile_workload",
        "description": "Measure a PyTorch matmul, conv2d, or attention workload and classify it with a roofline model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workload": {"type": "string", "enum": ["matmul", "conv2d", "attention"]},
                "device": {"type": "string", "default": "cuda"},
                "dtype": {"type": "string", "default": "float16"},
                "mode": {"type": "string", "enum": ["eager", "compile"], "default": "eager"},
                "compile_mode": {"type": "string", "enum": ["default", "reduce-overhead", "max-autotune"], "default": "default"},
                "warmup": {"type": "integer", "minimum": 0, "default": 5},
                "iterations": {"type": "integer", "minimum": 1, "default": 20},
                "dimensions": {"type": "object", "additionalProperties": {"type": "integer"}},
                "peak_tflops": {"type": "number", "default": 19.5},
                "bandwidth_gbps": {"type": "number", "default": 936.0},
            },
            "required": ["workload"],
        },
    },
    {
        "name": "autotune_workload",
        "description": "Benchmark eager, torch.compile modes, targeted layouts/fusions, and optional Triton tile configurations; return the fastest measured candidate.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workload": {"type": "string", "enum": ["matmul", "conv2d", "attention"]},
                "device": {"type": "string", "default": "cuda"},
                "dtype": {"type": "string", "default": "float16"},
                "warmup": {"type": "integer", "minimum": 0, "default": 5},
                "iterations": {"type": "integer", "minimum": 1, "default": 20},
                "dimensions": {"type": "object", "additionalProperties": {"type": "integer"}},
                "compile_modes": {"type": "array", "items": {"type": "string"}},
                "tile_sizes": {"type": "array", "items": {"type": "integer"}},
                "include_triton": {"type": "boolean", "default": True},
                "peak_tflops": {"type": "number", "default": 19.5},
                "bandwidth_gbps": {"type": "number", "default": 936.0},
            },
            "required": ["workload"],
        },
    },
    {
        "name": "tile_size_sweep",
        "description": "Estimate padding utilization for a matrix shape across tile sizes without requiring a GPU.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "m": {"type": "integer", "default": 4096},
                "n": {"type": "integer", "default": 4096},
                "k": {"type": "integer", "default": 4096},
                "dtype": {"type": "string", "default": "float16"},
                "tile_sizes": {"type": "array", "items": {"type": "integer"}},
            },
        },
    },
    {
        "name": "diagnose_pytorch",
        "description": "Run profiling and autotuning across matmul, conv2d, and attention, then emit JSON, CSV, and Markdown performance reports.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "device": {"type": "string", "default": "cuda"},
                "dtype": {"type": "string", "default": "float16"},
                "output_dir": {"type": "string", "default": "reports/latest"},
                "warmup": {"type": "integer", "minimum": 0, "default": 5},
                "iterations": {"type": "integer", "minimum": 1, "default": 20},
                "compile_modes": {"type": "array", "items": {"type": "string"}},
                "tile_sizes": {"type": "array", "items": {"type": "integer"}},
                "include_triton": {"type": "boolean", "default": True},
                "peak_tflops": {"type": "number", "default": 19.5},
                "bandwidth_gbps": {"type": "number", "default": 936.0},
            },
        },
    },
]


def _roofline(arguments: dict[str, Any]) -> HardwareRoofline:
    return HardwareRoofline(
        peak_tflops=float(arguments.get("peak_tflops", 19.5)),
        bandwidth_gbps=float(arguments.get("bandwidth_gbps", 936.0)),
    )


def _dimensions(arguments: dict[str, Any]) -> dict[str, int]:
    dimensions = arguments.get("dimensions", {})
    unknown = set(dimensions) - DIMENSION_NAMES
    if unknown:
        raise ValueError(f"Unknown dimensions: {', '.join(sorted(unknown))}")
    return {name: int(value) for name, value in dimensions.items()}


def _tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "structuredContent": payload,
    }


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    device = str(arguments.get("device", "cuda"))
    dtype = str(arguments.get("dtype", "float16"))
    warmup = int(arguments.get("warmup", 5))
    iterations = int(arguments.get("iterations", 20))

    if name == "profile_workload":
        workload = str(arguments["workload"])
        mode = str(arguments.get("mode", "eager"))
        compile_mode = str(arguments.get("compile_mode", "default")) if mode == "compile" else None
        spec = get_workload(workload, dtype=dtype, **_dimensions(arguments))
        result = profile_workload(
            spec,
            device=device,
            mode=mode,
            warmup=warmup,
            iterations=iterations,
            roofline=_roofline(arguments),
            compile_mode=compile_mode,
            configuration=f"compile/{compile_mode}" if compile_mode else mode,
        )
        return _tool_result(asdict(result))

    if name == "autotune_workload":
        tuning = run_autotune(
            str(arguments["workload"]),
            device=device,
            dtype=dtype,
            dimensions=_dimensions(arguments),
            warmup=warmup,
            iterations=iterations,
            roofline=_roofline(arguments),
            compile_modes=arguments.get(
                "compile_modes", ["default", "reduce-overhead", "max-autotune"]
            ),
            tile_sizes=arguments.get("tile_sizes", [32, 64, 128]),
            include_triton=bool(
                arguments.get("include_triton", device.startswith("cuda"))
            ),
        )
        return _tool_result(tuning.to_dict())

    if name == "tile_size_sweep":
        m = int(arguments.get("m", 4096))
        n = int(arguments.get("n", 4096))
        k = int(arguments.get("k", 4096))
        estimate = matmul_estimate(m, n, k, dtype)
        payload = {
            "shape": {"m": m, "n": n, "k": k},
            "arithmetic_intensity": estimate.arithmetic_intensity,
            "tiles": [
                tile_utilization(m, n, k, int(tile))
                for tile in arguments.get("tile_sizes", [32, 64, 128])
            ],
        }
        return _tool_result(payload)

    if name == "diagnose_pytorch":
        tunings = [
            run_autotune(
                workload,
                device=device,
                dtype=dtype,
                warmup=warmup,
                iterations=iterations,
                roofline=_roofline(arguments),
                compile_modes=arguments.get(
                    "compile_modes", ["default", "reduce-overhead", "max-autotune"]
                ),
                tile_sizes=arguments.get("tile_sizes", [32, 64, 128]),
                include_triton=bool(arguments.get("include_triton", device.startswith("cuda")))
                and workload == "matmul",
            )
            for workload in ("matmul", "conv2d", "attention")
        ]
        payload = diagnosis_payload(tunings)
        output_dir = Path(str(arguments.get("output_dir", "reports/latest")))
        output_dir.mkdir(parents=True, exist_ok=True)
        for tuning in tunings:
            write_json_payload(tuning.to_dict(), output_dir / f"{tuning.workload}.json")
            write_csv(tuning.results, output_dir / f"{tuning.workload}.csv")
        write_json_payload(payload, output_dir / "diagnosis.json")
        (output_dir / "diagnosis.md").write_text(
            render_diagnosis_markdown(payload), encoding="utf-8"
        )
        payload["output_dir"] = str(output_dir.resolve())
        return _tool_result(payload)

    raise ValueError(f"Unknown tool: {name}")


def dispatch(method: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    params = params or {}
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        return call_tool(str(params["name"]), dict(params.get("arguments", {})))
    if method.startswith("notifications/"):
        return None
    raise ValueError(f"Unsupported MCP method: {method}")


def main() -> None:
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(raw_line)
            result = dispatch(str(request["method"]), request.get("params"))
            if "id" not in request or result is None:
                continue
            response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"},
            }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
