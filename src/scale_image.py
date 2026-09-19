import cv2
import numpy as np


def _average_marker_spacing(points: list[tuple[float, float]]) -> float:
    """
    Average pixel distance between neighboring marker points.
    """

    if len(points) < 2:
        raise ValueError("At least two marker points are required.")

    pts = np.asarray(points, dtype=np.float64)

    distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)

    return float(np.mean(distances))


def estimate_pixels_per_mm(
    magenta_points: list[tuple[float, float]],
    cyan_points: list[tuple[float, float]],
    marker_spacing_mm: float = 50.0,
) -> float:
    """
    Estimate the current image scale from the marker spacing.

    Example:
    marker spacing = 50 mm
    average pixel spacing = 200 px

    -> 4 px/mm
    """

    spacings_px = []

    if len(magenta_points) >= 2:
        magenta_spacing = _average_marker_spacing(magenta_points)

        spacings_px.append(magenta_spacing)

    if len(cyan_points) >= 2:
        cyan_spacing = _average_marker_spacing(cyan_points)

        spacings_px.append(cyan_spacing)

    if not spacings_px:
        raise ValueError("Not enough calibration markers.")

    average_spacing_px = float(np.mean(spacings_px))

    pixels_per_mm = average_spacing_px / marker_spacing_mm

    return pixels_per_mm


def scale_image_to_pixels_per_mm(
    image_path: str,
    magenta_points: list[tuple[float, float]],
    cyan_points: list[tuple[float, float]],
    marker_spacing_mm: float = 50.0,
    target_pixels_per_mm: float = 4.0,
) -> tuple[np.ndarray, float]:
    """
    Uniformly scale an image to a desired metric resolution.

    Returns:
        scaled_image
        scale_factor
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    current_pixels_per_mm = estimate_pixels_per_mm(
        magenta_points,
        cyan_points,
        marker_spacing_mm,
    )

    scale_factor = target_pixels_per_mm / current_pixels_per_mm

    scaled_image = cv2.resize(
        image,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )

    print(f"Current scale: " f"{current_pixels_per_mm:.4f} px/mm")

    print(f"Target scale: " f"{target_pixels_per_mm:.4f} px/mm")

    print(f"Scale factor: " f"{scale_factor:.4f}")

    return scaled_image, scale_factor


def print_marker_spacing_report(
    name: str,
    points: list[tuple[float, float]],
    marker_spacing_mm: float = 50.0,
) -> None:

    pts = np.asarray(
        points,
        dtype=np.float64,
    )

    distances = np.linalg.norm(
        pts[1:] - pts[:-1],
        axis=1,
    )

    print()
    print(name)

    for i, distance_px in enumerate(distances):

        px_per_mm = distance_px / marker_spacing_mm

        print(f"{i} -> {i + 1}: " f"{distance_px:.2f} px " f"= {px_per_mm:.4f} px/mm")
