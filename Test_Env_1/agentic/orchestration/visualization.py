from __future__ import annotations

from pathlib import Path
from typing import Any


def export_langgraph_visualization(
    app: Any,
    name: str = "orchestration_graph",
    output_dir: str | Path = "outputs/graphs",
) -> dict[str, str | None]:
    """Export Mermaid text and attempt PNG output for a compiled LangGraph app."""

    target_directory = Path(output_dir)
    target_directory.mkdir(parents=True, exist_ok=True)

    mermaid_path = target_directory / f"{name}.mmd"
    png_path = target_directory / f"{name}.png"

    graph = app.get_graph()
    mermaid_text = graph.draw_mermaid()
    mermaid_path.write_text(mermaid_text, encoding="utf-8")

    print("LangGraph mermaid diagram:")
    print(mermaid_text)

    exported_png: str | None = None
    try:
        graph.draw_mermaid_png(output_file_path=str(png_path))
        if png_path.exists():
            exported_png = str(png_path)
            print(f"LangGraph PNG export: {exported_png}")
    except Exception as exc:
        print(
            "LangGraph PNG export unavailable; Mermaid text is still available. "
            f"Reason: {exc}"
        )

    return {
        "mermaid": str(mermaid_path),
        "png": exported_png,
    }
