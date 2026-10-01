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

# Flat-side detection
SAMPLE_FACE_COUNT = 120_000
FACE_CHUNK_SIZE = 150_000
NORMAL_BIN_SIZE = 0.03
NORMAL_TOLERANCE_DEG = 3.0
PLANE_BIN_MM = 0.25
PLANE_TOLERANCE_MM = 0.75

# Try several parallel planes instead of blindly accepting only one.
TOP_PLANE_CANDIDATES = 5

# Robustness against broken / imperfect meshes.
# Small gaps in the planar patch are bridged in the 2D mask.
MAX_GAP_MM = 1.5

# Small enclosed holes caused by missing triangles are filled.
# Large real openings, such as the big C-shaped cut-outs, stay open.
MAX_HOLE_AREA_MM2 = 100.0


# =========================================================
# BASIC HELPERS
# =========================================================


def clean_binary(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def triangle_normals_and_areas(
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]

    cross = np.cross(edge_1, edge_2)
    lengths = np.linalg.norm(cross, axis=1)

    areas = 0.5 * lengths

    normals = np.zeros_like(cross, dtype=np.float64)
    valid = lengths > 1e-12
    normals[valid] = cross[valid] / lengths[valid, None]

    return normals, areas


def canonicalize_normals(normals: np.ndarray) -> np.ndarray:
    """
    Treat n and -n as the same plane orientation.
    """
    result = normals.copy()

    dominant_axis = np.argmax(np.abs(result), axis=1)
    dominant_value = result[np.arange(len(result)), dominant_axis]
    sign = np.where(dominant_value < 0, -1.0, 1.0)

    result *= sign[:, None]
    return result


# =========================================================
# DETECT DOMINANT FLAT ORIENTATION
# =========================================================


def detect_dominant_flat_normal(mesh: trimesh.Trimesh) -> np.ndarray:
    face_count = len(mesh.faces)

    if face_count == 0:
        raise RuntimeError("Mesh has no faces.")

    sample_count = min(SAMPLE_FACE_COUNT, face_count)

    sample_indices = np.linspace(
        0,
        face_count - 1,
        sample_count,
        dtype=np.int64,
    )

    sample_faces = mesh.faces[sample_indices]
    sample_triangles = mesh.vertices[sample_faces]

    normals, areas = triangle_normals_and_areas(sample_triangles)

    valid = areas > 1e-12
    normals = normals[valid]
    areas = areas[valid]

    if len(normals) == 0:
        raise RuntimeError("Could not find valid sample triangles.")

    canonical = canonicalize_normals(normals)

    quantized = np.rint(canonical / NORMAL_BIN_SIZE).astype(np.int16)

    _, inverse = np.unique(
        quantized,
        axis=0,
        return_inverse=True,
    )

    area_per_bin = np.bincount(
        inverse,
        weights=areas,
    )

    best_bin = int(np.argmax(area_per_bin))
    selected = inverse == best_bin

    normal = np.average(
        canonical[selected],
        axis=0,
        weights=areas[selected],
    )

    length = np.linalg.norm(normal)

    if length <= 1e-12:
        raise RuntimeError("Could not determine flat-side normal.")

    return normal / length


# =========================================================
# FIND CANDIDATE PARALLEL PLANES
# =========================================================


def detect_plane_candidates(
    mesh: trimesh.Trimesh,
    normal: np.ndarray,
) -> list[dict]:
    """
    Find the strongest parallel planes for the dominant orientation.

    We keep several candidates because the STL may contain multiple
    large parallel flat regions. Later, each candidate is rasterized,
    split into connected patches, and scored independently.
    """

    face_count = len(mesh.faces)
    cos_limit = np.cos(np.deg2rad(NORMAL_TOLERANCE_DEG))

    # bin_id -> [area_sum, weighted_offset_sum]
    plane_bins: dict[int, list[float]] = {}

    for start in range(0, face_count, FACE_CHUNK_SIZE):
        end = min(start + FACE_CHUNK_SIZE, face_count)

        faces = mesh.faces[start:end]
        triangles = mesh.vertices[faces]

        normals, areas = triangle_normals_and_areas(triangles)

        alignment = np.abs(normals @ normal)
        keep = (areas > 1e-12) & (alignment >= cos_limit)

        if not np.any(keep):
            continue

        kept_triangles = triangles[keep]
        kept_areas = areas[keep]
        centers = np.mean(kept_triangles, axis=1)
        offsets = centers @ normal

        bin_ids = np.rint(offsets / PLANE_BIN_MM).astype(np.int64)

        for bin_id in np.unique(bin_ids):
            in_bin = bin_ids == bin_id

            area_sum = float(np.sum(kept_areas[in_bin]))
            weighted_offset_sum = float(np.sum(kept_areas[in_bin] * offsets[in_bin]))

            key = int(bin_id)

            if key not in plane_bins:
                plane_bins[key] = [0.0, 0.0]

            plane_bins[key][0] += area_sum
            plane_bins[key][1] += weighted_offset_sum

    if not plane_bins:
        raise RuntimeError("No nearly-planar faces found.")

    ranked = sorted(
        plane_bins.items(),
        key=lambda item: item[1][0],
        reverse=True,
    )[:TOP_PLANE_CANDIDATES]

    candidates = []

    for bin_id, (area_sum, weighted_sum) in ranked:
        candidates.append(
            {
                "bin_id": bin_id,
                "raw_area_mm2": area_sum,
                "offset_mm": weighted_sum / area_sum,
            }
        )

    return candidates


# =========================================================
# PLANE COORDINATE SYSTEM
# =========================================================


def build_plane_basis(
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build two orthonormal axes in the plane.

    Because U and V are orthonormal, the silhouette cannot be
    stretched differently in X and Y.
    """

    world_axes = np.eye(3, dtype=np.float64)

    helper = world_axes[np.argmin(np.abs(world_axes @ normal))]

    axis_u = np.cross(normal, helper)
    axis_u /= np.linalg.norm(axis_u)

    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)

    return axis_u, axis_v


# =========================================================
# ITERATE TRIANGLES ON ONE CANDIDATE PLANE
# =========================================================


def iter_plane_triangles(
    mesh: trimesh.Trimesh,
    normal: np.ndarray,
    plane_offset: float,
):
    face_count = len(mesh.faces)
    cos_limit = np.cos(np.deg2rad(NORMAL_TOLERANCE_DEG))

    for start in range(0, face_count, FACE_CHUNK_SIZE):
        end = min(start + FACE_CHUNK_SIZE, face_count)

        faces = mesh.faces[start:end]
        triangles = mesh.vertices[faces]

        normals, areas = triangle_normals_and_areas(triangles)
        alignment = np.abs(normals @ normal)

        centers = np.mean(triangles, axis=1)
        offsets = centers @ normal

        keep = (
            (areas > 1e-12)
            & (alignment >= cos_limit)
            & (np.abs(offsets - plane_offset) <= PLANE_TOLERANCE_MM)
        )

        if np.any(keep):
            yield triangles[keep], areas[keep]


# =========================================================
# MASK REPAIR / CONNECTED PATCH SELECTION
# =========================================================


def bridge_small_gaps(mask: np.ndarray) -> np.ndarray:
    gap_px = max(
        0,
        int(round(MAX_GAP_MM * PIXELS_PER_MM)),
    )

    if gap_px <= 0:
        return mask

    kernel_size = 2 * gap_px + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )


def keep_largest_connected_patch(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if count <= 1:
        return np.zeros_like(mask)

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(component_areas))

    result = np.zeros_like(mask)
    result[labels == largest_label] = 255

    return result


def fill_small_internal_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill only small black components fully enclosed by the white patch.
    Background touching the image edge is never filled.
    """

    result = clean_binary(mask)

    inverted = (result == 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted,
        connectivity=8,
    )

    if count <= 1:
        return result

    max_hole_px = int(round(MAX_HOLE_AREA_MM2 * PIXELS_PER_MM * PIXELS_PER_MM))

    border_labels = set(
        np.unique(
            np.concatenate(
                [
                    labels[0, :],
                    labels[-1, :],
                    labels[:, 0],
                    labels[:, -1],
                ]
            )
        ).tolist()
    )

    for label in range(1, count):
        if label in border_labels:
            continue

        area_px = int(stats[label, cv2.CC_STAT_AREA])

        if area_px <= max_hole_px:
            result[labels == label] = 255

    return result


def repair_and_select_patch(mask: np.ndarray) -> np.ndarray:
    mask = clean_binary(mask)
    mask = bridge_small_gaps(mask)
    mask = keep_largest_connected_patch(mask)
    mask = fill_small_internal_holes(mask)
    return clean_binary(mask)


# =========================================================
# RASTERIZE ONE PLANE CANDIDATE
# =========================================================


def rasterize_plane_candidate(
    mesh: trimesh.Trimesh,
    normal: np.ndarray,
    plane_offset: float,
    pixels_per_mm: float = PIXELS_PER_MM,
    padding_mm: float = PADDING_MM,
) -> tuple[np.ndarray, dict]:

    axis_u, axis_v = build_plane_basis(normal)
    origin = normal * plane_offset

    min_uv = np.array([np.inf, np.inf], dtype=np.float64)
    max_uv = np.array([-np.inf, -np.inf], dtype=np.float64)

    selected_faces = 0
    selected_triangle_area = 0.0

    # Pass 1: projected bounds
    for triangles, areas in iter_plane_triangles(
        mesh,
        normal,
        plane_offset,
    ):
        relative = triangles - origin
        u = relative @ axis_u
        v = relative @ axis_v
        uv = np.stack([u, v], axis=-1)

        points = uv.reshape(-1, 2)

        min_uv = np.minimum(
            min_uv,
            np.min(points, axis=0),
        )

        max_uv = np.maximum(
            max_uv,
            np.max(points, axis=0),
        )

        selected_faces += len(triangles)
        selected_triangle_area += float(np.sum(areas))

    if selected_faces == 0 or not np.all(np.isfinite(min_uv)):
        raise RuntimeError("Candidate plane has no usable triangles.")

    width_mm_raw = float(max_uv[0] - min_uv[0])
    height_mm_raw = float(max_uv[1] - min_uv[1])

    padding_px = int(round(padding_mm * pixels_per_mm))

    width_px = max(
        1,
        int(np.ceil(width_mm_raw * pixels_per_mm)) + 2 * padding_px,
    )

    height_px = max(
        1,
        int(np.ceil(height_mm_raw * pixels_per_mm)) + 2 * padding_px,
    )

    raw_mask = np.zeros(
        (height_px, width_px),
        dtype=np.uint8,
    )

    # Pass 2: rasterize coplanar triangles
    for triangles, _ in iter_plane_triangles(
        mesh,
        normal,
        plane_offset,
    ):
        relative = triangles - origin
        u = relative @ axis_u
        v = relative @ axis_v
        uv = np.stack([u, v], axis=-1)

        uv = (uv - min_uv) * pixels_per_mm
        uv[:, :, 0] += padding_px
        uv[:, :, 1] += padding_px

        # image Y points down
        uv[:, :, 1] = height_px - 1 - uv[:, :, 1]

        triangles_px = np.rint(uv).astype(np.int32)

        # cv2.fillPoly is much faster than one call per triangle.
        # Use moderate batches to avoid building a gigantic Python list.
        batch_size = 20_000

        for start in range(0, len(triangles_px), batch_size):
            batch = triangles_px[start : start + batch_size]

            contours = [triangle.reshape(-1, 1, 2) for triangle in batch]

            cv2.fillPoly(
                raw_mask,
                contours,
                color=255,
                lineType=cv2.LINE_8,
            )

    final_mask = repair_and_select_patch(raw_mask)

    ys, xs = np.where(final_mask > 0)

    if len(xs) == 0:
        raise RuntimeError("No connected planar patch survived cleanup.")

    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())

    component_width_px = x1 - x0 + 1
    component_height_px = y1 - y0 + 1

    component_width_mm = component_width_px / pixels_per_mm
    component_height_mm = component_height_px / pixels_per_mm

    component_area_px = int(np.count_nonzero(final_mask))
    component_area_mm2 = component_area_px / (pixels_per_mm * pixels_per_mm)

    # Crop to selected component, then restore standard padding.
    cropped = final_mask[y0 : y1 + 1, x0 : x1 + 1]

    padded = cv2.copyMakeBorder(
        cropped,
        padding_px,
        padding_px,
        padding_px,
        padding_px,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    info = {
        "selected_faces": selected_faces,
        "selected_triangle_area_mm2": selected_triangle_area,
        "component_area_mm2": component_area_mm2,
        "width_mm": component_width_mm,
        "height_mm": component_height_mm,
        "width_px": padded.shape[1],
        "height_px": padded.shape[0],
    }

    return padded, info


# =========================================================
# MESH -> BEST FLAT-SIDE MASK
# =========================================================


def mesh_to_mask(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, dict]:

    normal = detect_dominant_flat_normal(mesh)
    candidates = detect_plane_candidates(mesh, normal)

    print(
        "Dominant flat normal: "
        f"[{normal[0]:.4f}, "
        f"{normal[1]:.4f}, "
        f"{normal[2]:.4f}]"
    )

    best_mask = None
    best_info = None
    best_score = -1.0

    for index, candidate in enumerate(candidates, start=1):
        try:
            mask, info = rasterize_plane_candidate(
                mesh,
                normal,
                candidate["offset_mm"],
            )
        except RuntimeError as exc:
            print(f"  Candidate {index}: skipped ({exc})")
            continue

        # Main selection criterion:
        # largest single connected planar patch after gap/hole repair.
        score = info["component_area_mm2"]

        print(
            f"  Candidate {index}: "
            f"plane={candidate['offset_mm']:.3f} mm | "
            f"patch={info['width_mm']:.2f} x "
            f"{info['height_mm']:.2f} mm | "
            f"area={info['component_area_mm2']:.1f} mm^2"
        )

        if score > best_score:
            best_score = score
            best_mask = mask
            best_info = {
                **info,
                "normal": normal,
                "plane_offset_mm": candidate["offset_mm"],
                "candidate_index": index,
            }

    if best_mask is None or best_info is None:
        raise RuntimeError("Could not detect a usable flat side.")

    return best_mask, best_info


# =========================================================
# PROCESS STL
# =========================================================


def process_stl(stl_path: Path):
    print()
    print("========================================")
    print(f"Processing: {stl_path.name}")
    print("========================================")

    mesh = trimesh.load_mesh(
        stl_path,
        process=True,
    )

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("STL did not load as a single Trimesh.")

    mask, info = mesh_to_mask(mesh)

    output_path = OUTPUT_DIR / f"{stl_path.stem}_mask.png"

    success = cv2.imwrite(
        str(output_path),
        mask,
    )

    if not success:
        raise RuntimeError(f"Could not save mask: {output_path}")

    normal = info["normal"]

    print(
        "Chosen flat normal: "
        f"[{normal[0]:.4f}, "
        f"{normal[1]:.4f}, "
        f"{normal[2]:.4f}]"
    )

    print(f"Chosen candidate: {info['candidate_index']}")

    print(
        "Final flat-side size: "
        f"{info['width_mm']:.2f} x "
        f"{info['height_mm']:.2f} mm"
    )

    print("Connected patch area: " f"{info['component_area_mm2']:.1f} mm^2")

    print("Mask size: " f"{info['width_px']} x " f"{info['height_px']} px")

    print(f"Saved: {output_path}")


# =========================================================
# MAIN
# =========================================================


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
