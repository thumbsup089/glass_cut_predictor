import os
import numpy as np
import svgwrite


def export_radial_cut_svg(radial_data, output_path, margin_mm=10.0):
    points = []

    for item in radial_data:

        r = item["radius_mm"]

        if not np.isfinite(r):
            print("Invalid radius skipped:", item)
            continue

        theta = np.deg2rad(item["angle_deg"])

        x = r * np.cos(theta)

        y = r * np.sin(theta)

        points.append((x, y))

    if len(points) < 3:
        raise ValueError("Not enough valid points " "for SVG contour.")

    xs = [p[0] for p in points]

    ys = [p[1] for p in points]

    min_x = min(xs)
    max_x = max(xs)

    min_y = min(ys)
    max_y = max(ys)

    width = max_x - min_x
    height = max_y - min_y

    svg_width = width + 2 * margin_mm

    svg_height = height + 2 * margin_mm

    if not np.isfinite(svg_width) or not np.isfinite(svg_height):
        raise ValueError("Invalid SVG dimensions.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{svg_width}mm", f"{svg_height}mm"),
        viewBox=(
            f"{min_x - margin_mm} "
            f"{min_y - margin_mm} "
            f"{svg_width} "
            f"{svg_height}"
        ),
    )

    dwg.add(dwg.polygon(points=points, fill="none", stroke="black", stroke_width=0.5))

    dwg.save()

    print(f"SVG gespeichert: " f"{output_path}")
