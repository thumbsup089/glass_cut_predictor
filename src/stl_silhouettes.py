from pathlib import Path

import cv2
import numpy as np
import trimesh

# =========================================================
# CONFIG
# =========================================================

BRAND_DIR = Path("data/Brandt001_14_09_2026")
STL_DIR = BRAND_DIR / "stls"
OUTPUT_DIR = BRAND_DIR / "processed" / "stl_masks"

PIXELS_PER_MM = 4.0
PADDING_MM = 5.0


# =========================================================
# HELPERS
# =========================================================


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    return mask


def mesh_to_mask(
    mesh: trimesh.Trimesh,
    pixels_per_mm: float = PIXELS_PER_MM,
    padding_mm: float = PADDING_MM,
) -> np.ndarray:
    """
    Projiziert das gesamte STL auf die XY-Ebene
    und rasterisiert daraus eine binäre Maske:

    schwarz = Hintergrund
    weiß    = STL-Form
    """

    vertices_xy = np.asarray(
        mesh.vertices[:, :2],
        dtype=np.float64,
    )

    if len(vertices_xy) == 0:
        raise RuntimeError("Mesh has no vertices.")

    min_xy = np.min(vertices_xy, axis=0)
    max_xy = np.max(vertices_xy, axis=0)

    width_mm = float(max_xy[0] - min_xy[0])
    height_mm = float(max_xy[1] - min_xy[1])

    width_px = max(
        1,
        int(np.ceil((width_mm + 2 * padding_mm) * pixels_per_mm)),
    )

    height_px = max(
        1,
        int(np.ceil((height_mm + 2 * padding_mm) * pixels_per_mm)),
    )

    mask = np.zeros(
        (height_px, width_px),
        dtype=np.uint8,
    )

    vertices_px = (vertices_xy - min_xy + padding_mm) * pixels_per_mm

    vertices_px = np.round(vertices_px).astype(np.int32)

    for face in mesh.faces:
        triangle = vertices_px[face]

        cv2.fillConvexPoly(
            mask,
            triangle,
            color=255,
            lineType=cv2.LINE_8,
        )

    mask = clean_mask(mask)

    return mask


def process_stl(stl_path: Path):
    print(f"Processing: {stl_path.name}")

    mesh = trimesh.load_mesh(
        stl_path,
        process=True,
    )

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("STL did not load as a single Trimesh.")

    mask = mesh_to_mask(mesh)

    output_path = OUTPUT_DIR / f"{stl_path.stem}_mask.png"

    success = cv2.imwrite(
        str(output_path),
        mask,
    )

    if not success:
        raise RuntimeError(f"Could not save mask: {output_path}")

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stl_files = sorted(
        [
            path
            for path in STL_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == ".stl"
        ],
        key=lambda path: path.name.lower(),
    )

    if not stl_files:
        raise RuntimeError(f"No STL files found in {STL_DIR}")

    for stl_path in stl_files:
        process_stl(stl_path)


if __name__ == "__main__":
    main()
