"""Build a .ipynb from a percent-format .py source.

Cells are delimited by ``# %%`` (code) and ``# %% [markdown]`` (markdown, whose
body is written as ``# `` comments).  Keeping the tutorials in plain .py under
version control makes them reviewable as diffs; the .ipynb is generated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def parse(source: str):
    cells, kind, buffer = [], "code", []

    def flush():
        if not buffer:
            return
        text = "\n".join(buffer).strip("\n")
        if not text.strip():
            return
        if kind == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else ("" if line.strip() == "#" else line)
                for line in text.splitlines()
            )
            cells.append(nbformat.v4.new_markdown_cell(body.strip("\n")))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            buffer = []
            kind = "markdown" if "[markdown]" in line else "code"
            continue
        buffer.append(line)
    flush()
    return cells


def main(src: str, dst: str) -> None:
    nb = nbformat.v4.new_notebook(cells=parse(Path(src).read_text()))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    nbformat.write(nb, dst)
    print(f"wrote {dst} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
