from types import SimpleNamespace

from torch_tile_profiler import mcp_server


dispatch = mcp_server.dispatch


def test_mcp_lists_all_benchmark_tools() -> None:
    result = dispatch("tools/list")
    assert result is not None
    names = {tool["name"] for tool in result["tools"]}
    assert names == {
        "profile_workload",
        "autotune_workload",
        "tile_size_sweep",
        "diagnose_pytorch",
    }


def test_mcp_tile_sweep_is_available_without_torch_or_cuda() -> None:
    result = dispatch(
        "tools/call",
        {"name": "tile_size_sweep", "arguments": {"m": 130, "n": 128, "k": 64, "tile_sizes": [64]}},
    )
    assert result is not None
    assert result["structuredContent"]["tiles"][0]["tiles_m"] == 3


def test_mcp_autotune_forwards_configuration_and_returns_results(monkeypatch) -> None:
    captured = {}
    payload = {"workload": "matmul", "best_configuration": "compile/default"}

    def fake_run_autotune(workload, **kwargs):
        captured["workload"] = workload
        captured.update(kwargs)
        return SimpleNamespace(to_dict=lambda: payload)

    monkeypatch.setattr(mcp_server, "run_autotune", fake_run_autotune)
    result = dispatch(
        "tools/call",
        {
            "name": "autotune_workload",
            "arguments": {
                "workload": "matmul",
                "device": "cpu",
                "dimensions": {"m": 64, "n": 32, "k": 16},
                "compile_modes": ["default"],
                "tile_sizes": [32, 64],
                "include_triton": False,
            },
        },
    )

    assert result is not None
    assert result["structuredContent"] == payload
    assert captured["workload"] == "matmul"
    assert captured["dimensions"] == {"m": 64, "n": 32, "k": 16}
    assert captured["compile_modes"] == ["default"]
    assert captured["tile_sizes"] == [32, 64]
    assert captured["include_triton"] is False
