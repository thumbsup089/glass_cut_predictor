import numpy as np


def bilinear_sample(heightmap, x, y):
    h, w = heightmap.shape

    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return None

    x0 = int(np.floor(x))
    x1 = x0 + 1

    y0 = int(np.floor(y))
    y1 = y0 + 1

    values = np.array(
        [heightmap[y0, x0], heightmap[y0, x1], heightmap[y1, x0], heightmap[y1, x1]]
    )

    if not np.all(np.isfinite(values)):
        return None

    dx = x - x0
    dy = y - y0

    z00, z10, z01, z11 = values

    z0 = z00 * (1 - dx) + z10 * dx

    z1 = z01 * (1 - dx) + z11 * dx

    return z0 * (1 - dy) + z1 * dy


def calculate_radial_cut(
    heightmap, pixel_size_mm, center, angle_count=360, step_px=0.5
):
    cx, cy = center

    results = []

    for angle_deg in range(angle_count):

        theta = np.deg2rad(angle_deg)

        direction_x = np.cos(theta)

        direction_y = np.sin(theta)

        previous_x = cx
        previous_y = cy

        previous_z = bilinear_sample(heightmap, previous_x, previous_y)

        if previous_z is None:
            raise ValueError("Radial origin lies outside " "the valid heightmap.")

        surface_length_mm = 0.0

        r = step_px

        while True:

            x = cx + r * direction_x

            y = cy + r * direction_y

            z = bilinear_sample(heightmap, x, y)

            if z is None:
                break

            dx_mm = (x - previous_x) * pixel_size_mm

            dy_mm = (y - previous_y) * pixel_size_mm

            dz_mm = z - previous_z

            segment_length = np.sqrt(dx_mm**2 + dy_mm**2 + dz_mm**2)

            surface_length_mm += segment_length

            previous_x = x
            previous_y = y
            previous_z = z

            r += step_px

        results.append({"angle_deg": angle_deg, "radius_mm": surface_length_mm})

    radii = [item["radius_mm"] for item in results]

    print(
        f"Radial cut: "
        f"{len(results)} directions, "
        f"radii "
        f"{min(radii):.3f}.."
        f"{max(radii):.3f} mm"
    )

    return results
