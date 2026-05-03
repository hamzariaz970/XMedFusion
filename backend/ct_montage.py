from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image


def _sample_indices(total: int, limit: int, trim_top: float = 0.15, trim_bottom: float = 0.10) -> list[int]:
    if total <= 0:
        return []

    # If the clinician already supplied a small curated slice set, preserve it
    # exactly instead of trimming away informative views and padding blanks.
    if total <= limit:
        return list(range(total))

    start = int(total * trim_top)
    end = int(total * (1.0 - trim_bottom))
    if end <= start:
        start, end = 0, total

    usable = list(range(start, end))
    if not usable:
        usable = list(range(total))

    if len(usable) <= limit:
        return usable

    if limit == 1:
        return [usable[len(usable) // 2]]

    positions = [round(i * (len(usable) - 1) / (limit - 1)) for i in range(limit)]
    return [usable[pos] for pos in positions]


def build_ct_montage(
    image_paths: Sequence[str | Path],
    *,
    rows: int = 4,
    cols: int = 4,
    tile_size: tuple[int, int] = (256, 256),
) -> tuple[Image.Image, dict]:
    normalized_paths = [Path(path) for path in image_paths]
    if not normalized_paths:
        raise ValueError("At least one CT image path is required.")

    total_cells = rows * cols
    selected_indices = _sample_indices(len(normalized_paths), total_cells)
    selected_paths = [normalized_paths[idx] for idx in selected_indices]

    tile_w, tile_h = tile_size
    montage = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(0, 0, 0))
    slice_cells = []

    for cell_idx in range(total_cells):
        col = cell_idx % cols
        row = cell_idx // cols
        x = col * tile_w
        y = row * tile_h

        if cell_idx < len(selected_paths):
            source_path = selected_paths[cell_idx]
            with Image.open(source_path) as img:
                tile = img.convert("RGB").resize((tile_w, tile_h))
            montage.paste(tile, (x, y))
            slice_cells.append(
                {
                    "cell_index": cell_idx + 1,
                    "source_path": str(source_path),
                    "source_filename": source_path.name,
                    "source_order_index": selected_indices[cell_idx] + 1,
                    "row": row,
                    "col": col,
                }
            )
        else:
            slice_cells.append(
                {
                    "cell_index": cell_idx + 1,
                    "source_path": None,
                    "source_filename": None,
                    "source_order_index": None,
                    "row": row,
                    "col": col,
                }
            )

    metadata = {
        "rows": rows,
        "cols": cols,
        "tile_width": tile_w,
        "tile_height": tile_h,
        "selected_slice_count": len(selected_paths),
        "source_image_count": len(normalized_paths),
        "slice_cells": slice_cells,
    }
    return montage, metadata
