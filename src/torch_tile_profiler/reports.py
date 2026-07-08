from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

from .profiler import ProfileResult


def result_to_dict(result: ProfileResult | dict[str, object]) -> dict[str, object]:
    if is_dataclass(result):
        return asdict(result)
    return dict(result)


def write_json(results: Iterable[ProfileResult | dict[str, object]], path: str | Path) -> None:
    rows = [result_to_dict(result) for result in results]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_csv(results: Iterable[ProfileResult | dict[str, object]], path: str | Path) -> None:
    rows = [result_to_dict(result) for result in results]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "workload",
        "mode",
        "device",
        "dtype",
        "time_ms",
        "flops",
        "memory_bytes",
        "arithmetic_intensity",
        "achieved_gflops",
        "bottleneck",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def print_table(results: Iterable[ProfileResult | dict[str, object]]) -> None:
    rows = []
    for result in results:
        row = result_to_dict(result)
        rows.append(
            [
                str(row["workload"]),
                str(row["mode"]),
                str(row["device"]),
                f"{float(row['time_ms']):.2f}",
                f"{float(row['achieved_gflops']):.1f}",
                f"{float(row['arithmetic_intensity']):.2f}",
                str(row["bottleneck"]),
            ]
        )

    headers = ["Workload", "Mode", "Device", "Time (ms)", "GFLOP/s", "AI (F/B)", "Bottleneck"]
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="PyTorch Compute Tile Utilization Profiler")
        for column in headers:
            table.add_column(column)
        for row in rows:
            table.add_row(*row)
        Console().print(table)
    except ImportError:
        widths = [len(header) for header in headers]
        for row in rows:
            widths = [max(width, len(value)) for width, value in zip(widths, row)]
        print("PyTorch Compute Tile Utilization Profiler")
        print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
        print("-+-".join("-" * width for width in widths))
        for row in rows:
            print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))
