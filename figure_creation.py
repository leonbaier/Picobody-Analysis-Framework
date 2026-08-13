from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_composite_figure(images: dict[str, Path], layout: str, output_file: Path, panel_label_size: int = 80,
        figure_width_px: int = 2400,
):
    """
    Create a composite figure from existing image files.

    Example
    -------
    images = {
        "A": Path("schema.png"),
        "B": Path("rmsd.png"),
        "C": Path("rmsf.png"),
    }

    layout = '''
    AA
    BC
    '''
    """

    layout_rows = [row.strip()
        for row in layout.strip().splitlines()
        if row.strip()
    ]

    n_rows = len(layout_rows)
    n_cols = len(layout_rows[0])

    for row in layout_rows:
        if len(row) != n_cols:
            raise ValueError("All layout rows must have equal length.")

    cell_width = figure_width_px // n_cols
    cell_height = cell_width
    figure_height = cell_height * n_rows

    final_image = Image.new("RGB", (figure_width_px, figure_height),"white",)

    try:
        font = ImageFont.truetype("arialbd.ttf", panel_label_size,)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(final_image)
    labels = sorted(set("".join(layout_rows)))

    for label in labels:

        if label not in images:
            raise KeyError(f"No image provided for panel '{label}'.")

        positions = []

        for row_idx, row in enumerate(layout_rows):
            for col_idx, value in enumerate(row):

                if value == label:
                    positions.append((row_idx, col_idx))

        rows = [r for r, _ in positions]
        cols = [c for _, c in positions]

        min_row = min(rows)
        max_row = max(rows)

        min_col = min(cols)
        max_col = max(cols)

        panel_x = min_col * cell_width
        panel_y = min_row * cell_height

        panel_width = ((max_col - min_col + 1) * cell_width)
        panel_height = ((max_row - min_row + 1) * cell_height)

        img = Image.open(images[label]).convert("RGB")

        scale = min(
            panel_width / img.width,
            panel_height / img.height,)

        new_width = int(img.width * scale)
        new_height = int(img.height * scale)

        img = img.resize((new_width, new_height), Image.LANCZOS,)

        paste_x = (panel_x + (panel_width - new_width) // 2)
        paste_y = (panel_y + (panel_height - new_height) // 2)
        final_image.paste(img, (paste_x, paste_y),)

        draw.text(
            (
                panel_x + 20,
                panel_y + 20,
            ),
            label,
            fill="black",
            font=font,
        )

    output_file.parent.mkdir(parents=True, exist_ok=True,)
    final_image.save(output_file)

    print(f"[Figure Creation] Saved: {output_file}")