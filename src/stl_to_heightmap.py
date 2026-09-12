import numpy as np
import trimesh


def fit_plane(points):
    """
    Fits:
        z = a*x + b*y + c
    """
    A = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])

    z = points[:, 2]

    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    return coeffs


def plane_z(x, y, plane):
    a, b, c = plane
    return a * x + b * y + c


def stl_to_heightmap(stl_path, resolution_mm=0.5):
    """
    STL -> Heightmap

    Assumptions:
    - STL coordinates are preserved
    - STL origin x=0, y=0 is the radial-cut origin
    - one surface is approximately planar
    - the other surface is the actual glass-contact surface
    """

    mesh = trimesh.load_mesh(stl_path)

    if mesh.is_empty:
        raise ValueError("STL could not be loaded or is empty.")

    min_x, min_y, min_z = mesh.bounds[0]
    max_x, max_y, max_z = mesh.bounds[1]

    print(
        f"STL bounds: "
        f"X {min_x:.3f}..{max_x:.3f}, "
        f"Y {min_y:.3f}..{max_y:.3f}, "
        f"Z {min_z:.3f}..{max_z:.3f} mm"
    )

    # Make sure x=0 and y=0 exist exactly in the raster
    xs_negative = np.arange(0, min_x - resolution_mm, -resolution_mm)[::-1]

    xs_positive = np.arange(resolution_mm, max_x + resolution_mm, resolution_mm)

    xs = np.concatenate([xs_negative, xs_positive])

    ys_negative = np.arange(0, min_y - resolution_mm, -resolution_mm)[::-1]

    ys_positive = np.arange(resolution_mm, max_y + resolution_mm, resolution_mm)

    ys = np.concatenate([ys_negative, ys_positive])

    X, Y = np.meshgrid(xs, ys)

    ray_start_z = max_z + 10.0

    origins = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, ray_start_z)])

    directions = np.tile(np.array([0.0, 0.0, -1.0]), (len(origins), 1))

    print(
        f"Grid dimensions: "
        f"{len(xs)} x {len(ys)}; "
        f"number of rays: {len(origins):,}"
    )

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )

    hits_by_ray = {}

    for location, ray_idx in zip(locations, index_ray):
        hits_by_ray.setdefault(ray_idx, []).append(location)

    # Collect lower and upper intersection envelopes
    lower_points = []
    upper_points = []

    for ray_idx, hits in hits_by_ray.items():

        if len(hits) < 2:
            continue

        hits = np.asarray(hits)

        z_values = hits[:, 2]

        lower_idx = np.argmin(z_values)
        upper_idx = np.argmax(z_values)

        lower_points.append(hits[lower_idx])

        upper_points.append(hits[upper_idx])

    lower_points = np.asarray(lower_points)
    upper_points = np.asarray(upper_points)

    if len(lower_points) < 3 or len(upper_points) < 3:
        raise ValueError(
            "Not enough ray intersections " "to identify the reference plane."
        )

    lower_plane = fit_plane(lower_points)

    upper_plane = fit_plane(upper_points)

    lower_residuals = np.abs(
        lower_points[:, 2]
        - plane_z(lower_points[:, 0], lower_points[:, 1], lower_plane)
    )

    upper_residuals = np.abs(
        upper_points[:, 2]
        - plane_z(upper_points[:, 0], upper_points[:, 1], upper_plane)
    )

    lower_median = np.median(lower_residuals)

    upper_median = np.median(upper_residuals)

    if upper_median < lower_median:
        reference_plane = upper_plane
        plane_name = "upper envelope"
    else:
        reference_plane = lower_plane
        plane_name = "lower envelope"

    print(
        f"Fitted reference plane ({plane_name}): "
        f"z = {reference_plane[0]:.7g}*x + "
        f"{reference_plane[1]:.7g}*y + "
        f"{reference_plane[2]:.7g}"
    )

    print(
        f"Plane median residuals: "
        f"lower={lower_median:.4g} mm, "
        f"upper={upper_median:.4g} mm"
    )

    heightmap_flat = np.full(X.size, np.nan, dtype=np.float64)

    for ray_idx, hits in hits_by_ray.items():

        if len(hits) == 0:
            continue

        hits = np.asarray(hits)

        x = origins[ray_idx, 0]
        y = origins[ray_idx, 1]

        z_ref = plane_z(x, y, reference_plane)

        z_values = hits[:, 2]

        distances = np.abs(z_values - z_ref)

        form_index = np.argmax(distances)

        heightmap_flat[ray_idx] = z_values[form_index]

    heightmap = heightmap_flat.reshape(X.shape)

    cx = int(np.argmin(np.abs(xs)))

    cy = int(np.argmin(np.abs(ys)))

    print(
        f"Origin index: "
        f"(cx={cx}, cy={cy}); "
        f"z at origin: "
        f"{heightmap[cy, cx]}"
    )

    if not np.isfinite(heightmap[cy, cx]):
        raise ValueError("No valid surface found at " "STL origin x=0, y=0.")

    valid_percent = np.count_nonzero(np.isfinite(heightmap)) / heightmap.size * 100.0

    print(f"Valid heightmap cells: " f"{valid_percent:.2f}%")

    return (heightmap, xs, ys, (cx, cy))
