"""
Build the Colab .ipynb file from NAVDRIFT0_Training.py cell definitions.
Run: python notebooks/build_notebook.py
"""

import json
from pathlib import Path
from NAVDRIFT0_Training import NOTEBOOK_CELLS


def build_ipynb(cells, output_path: Path) -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "A100",
                "include_colab_link": True,
            },
        },
        "cells": [],
    }

    # Header markdown
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# NAVDRIFT-0: AI-ML Dead Reckoning for Seamless Navigation\n",
            "\n",
            "**SIH 2026 | Problem Statement 26168 | ISRO**\n",
            "\n",
            "Run cells in order. Cell 0 (anti-disconnect) should stay running throughout.\n",
            "All checkpoints save to Google Drive automatically.\n",
        ],
        "id": "header",
    })

    cell_labels = [
        "Cell 0: Anti-Disconnect",
        "Cell 1: Connect Google Drive",
        "Cell 2: Install Dependencies",
        "Cell 3: Clone Repository",
        "Cell 4: Download IO-VNBD Dataset",
        "Cell 5: Parse and Preprocess Data",
        "Cell 6: Train DRIFT-Former",
        "Cell 7: Train NavIC Motion Prior VAE",
        "Cell 8: Export to ONNX + Quantize",
        "Cell 9: Evaluate Baselines vs NAVDRIFT-0",
        "Cell 10: Visualize Results",
    ]

    for i, (cell, label) in enumerate(zip(cells, cell_labels)):
        # Add markdown header
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {label}\n"],
            "id": f"md_{i}",
        })
        # Add code cell
        source_lines = [line + "\n" for line in cell["source"].splitlines()]
        if source_lines:
            source_lines[-1] = source_lines[-1].rstrip("\n")
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": f"cell_{i}"},
            "outputs": [],
            "source": source_lines,
            "id": f"code_{i}",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Notebook written: {output_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from NAVDRIFT0_Training import NOTEBOOK_CELLS

    out = Path(__file__).parent / "NAVDRIFT0_Training.ipynb"
    build_ipynb(NOTEBOOK_CELLS, out)
