from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont


def create_composite_figure(images: dict[str, Path], layout: str, output_file: Path, figure_width_px: int = 3000,
                            row_spacing: int = 20, col_spacing: int = 60, panel_label_size: int = 50,
                            panel_label_offset_x: int = 20, panel_label_offset_y: int = 25, right_column_label_offset: int = 0
):

    """
    Create a composite figure from existing image files.

    Each unique character in the layout corresponds to one panel.

    Example
    -------
    images = {
        "A": Path("plot_a.png"),
        "B": Path("plot_b.png"),
        "C": Path("plot_c.png"),
        "D": Path("plot_d.png"),
        "E": Path("plot_e.png"),
    }

    layout = '''
    AB
    CD
    EE
    '''

    Parameters
    ----------
    images : dict[str, Path]
        Mapping between panel labels and image files.

    layout : str
        Panel arrangement.

    output_file : Path
        Output image path.

    figure_width_px : int, default=3000
        Width of final figure.

    row_spacing : int, default=20
        Vertical spacing between rows.

    col_spacing : int, default=25
        Horizontal spacing between columns.

    panel_label_size : int, default=50
        Font size of panel labels.

    panel_label_offset: int, default=25
        Offset of panel labels.
    """

    layout_rows = [row.strip() for row in layout.strip().splitlines() if row.strip()]

    n_rows = len(layout_rows)
    n_cols = len(layout_rows[0])

    for row in layout_rows:
        if len(row) != n_cols:
            raise ValueError("All layout rows must have equal length.")

    cell_width = (figure_width_px - col_spacing * (n_cols - 1)) // n_cols

    try:
        font = ImageFont.truetype("arialbd.ttf", panel_label_size)
    except Exception:
        font = ImageFont.load_default()

    # --------------------------------------------------
    # determine panel geometry
    # --------------------------------------------------

    panels = {}
    labels = sorted(set("".join(layout_rows)))
    row_heights = [0.0] * n_rows  # Use float for accurate height distribution

    for label in labels:
        if label not in images:
            raise KeyError(f"No image provided for panel '{label}'.")

        img = Image.open(images[label]).convert("RGB")

        positions = []
        for r, row in enumerate(layout_rows):
            for c, value in enumerate(row):
                if value == label:
                    positions.append((r, c))

        rows = [r for r, _ in positions]
        cols = [c for _, c in positions]

        min_row = min(rows)
        max_row = max(rows)
        min_col = min(cols)
        max_col = max(cols)

        span_cols = max_col - min_col + 1

        panel_width = (span_cols * cell_width + (span_cols - 1) * col_spacing)

        scale = panel_width / img.width
        scaled_width = int(img.width * scale)
        scaled_height = int(img.height * scale)

        img = img.resize((scaled_width, scaled_height), Image.LANCZOS)

        panels[label] = {
            "image": img,
            "min_row": min_row,
            "max_row": max_row,
            "col": min_col,
            "width": scaled_width,
            "height": scaled_height,
        }

    # --------------------------------------------------
    # TWO-PASS ROW HEIGHT CALCULATION (fixes vertical spanning)
    # --------------------------------------------------

    # Pass 1: Set heights based strictly on panels that span exactly 1 row
    for label, panel in panels.items():
        if panel["min_row"] == panel["max_row"]:
            r = panel["min_row"]
            row_heights[r] = max(row_heights[r], panel["height"])

    # Pass 2: Distribute extra height required by multi-row panels (like AC / BC)
    for label, panel in panels.items():
        if panel["min_row"] < panel["max_row"]:
            span_rows = panel["max_row"] - panel["min_row"] + 1
            current_span_height = sum(row_heights[panel["min_row"]:panel["max_row"] + 1]) + row_spacing * (
                        span_rows - 1)

            if panel["height"] > current_span_height:
                # Add the missing height equally to all rows it spans
                deficit = panel["height"] - current_span_height
                add_per_row = deficit / span_rows
                for r in range(panel["min_row"], panel["max_row"] + 1):
                    row_heights[r] += add_per_row

    figure_height = int(sum(row_heights) + row_spacing * (n_rows - 1))

    final_image = Image.new("RGB", (figure_width_px, figure_height), "white")
    draw = ImageDraw.Draw(final_image)

    row_offsets = []
    current_y = 0

    for h in row_heights:
        row_offsets.append(int(current_y))
        current_y += h + row_spacing

    # --------------------------------------------------
    # place images
    # --------------------------------------------------

    for label, panel in panels.items():
        panel_x = (panel["col"] * (cell_width + col_spacing))
        panel_y = row_offsets[panel["min_row"]]

        final_image.paste(panel["image"], (panel_x, panel_y))

        # Fixed label offset logic based on actual panel column
        if panel["col"] == 0:
            label_x = max(5, panel_x - panel_label_offset_x)
        else:
            label_x = max(5, panel_x - panel_label_offset_x) + right_column_label_offset

        label_y = max(5, panel_y - panel_label_offset_y)

        draw.text((label_x, label_y), label, fill="black", font=font)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(output_file)

    print(f"[Figure Creation] Saved: {output_file}")


def export_single_panel_figures(figures_mapping: dict[str, Path], output_dir: Path):
    """
    Exports (copies) standalone figures (like MS PDFs) to the final thesis figure directory,
    renaming them according to the provided mapping.

    Parameters
    ----------
    figures_mapping : dict[str, Path]
        Mapping of target filename (e.g. "Figure_09_MS_V1.pdf") to the source file path.
    output_dir : Path
        Directory where the final thesis figures should be stored.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for fig_name, source_path in figures_mapping.items():
        if source_path is not None and source_path.exists():
            dest_path = output_dir / fig_name

            # Use shutil.copy2 to preserve metadata and allow repeated script execution
            shutil.copy2(source_path, dest_path)

            print(f"[Figure Creation] Saved standalone figure: {dest_path}")
        else:
            print(f"[Figure Creation] Skipping {fig_name}: Source file not found ({source_path})")