from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MarkerDetectionResult:
    magenta_points: list[tuple[float, float]]
    cyan_points: list[tuple[float, float]]

    # Alle grünen Pixel des eingezeichneten 90°-Winkels
    green_pixels: np.ndarray

    # Binärmasken, später nützlich für Kalibrierung / Debugging
    magenta_mask: np.ndarray
    cyan_mask: np.ndarray
    green_mask: np.ndarray


def _create_color_mask(
    image_bgr: np.ndarray, target_bgr: tuple[int, int, int], tolerance: int = 40
) -> np.ndarray:
    """
    Find pixels close to a target BGR color.

    GIMP anti-aliasing can create pixels that are not exactly
    #FF00FF / #00FFFF / #00FF00, therefore a tolerance is used.
    """

    image = image_bgr.astype(np.int16)
    target = np.array(target_bgr, dtype=np.int16)

    diff = np.abs(image - target)

    mask = np.all(diff <= tolerance, axis=2)

    return mask.astype(np.uint8) * 255


def _find_dot_centers(
    mask: np.ndarray,
    min_area: int = 30,
    max_area: int = 800,
) -> list[tuple[float, float]]:
    """
    Detect connected colored blobs and return their centroids.

    A 15 px diameter marker has roughly ~175 px area,
    so 30..800 leaves some tolerance.
    """

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    points = []

    # label 0 = background
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if min_area <= area <= max_area:
            x, y = centroids[label]
            points.append((float(x), float(y)))

    return points


def _sort_points_along_line(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Sort points along their principal direction using PCA.

    This means it does not matter whether the measuring tape
    is horizontal, vertical, diagonal, or perspective-skewed.
    """

    if len(points) <= 1:
        return points

    pts = np.asarray(points, dtype=np.float64)

    center = np.mean(pts, axis=0)
    centered = pts - center

    # Principal component of the point cloud
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    direction = vh[0]

    projection = centered @ direction

    order = np.argsort(projection)

    sorted_points = pts[order]

    return [(float(point[0]), float(point[1])) for point in sorted_points]


def detect_markers(
    image_path: str,
    color_tolerance: int = 40,
    min_dot_area: int = 30,
    max_dot_area: int = 800,
) -> MarkerDetectionResult:
    """
    Detect calibration annotations in an image.

    Expected colors:

    Magenta #FF00FF:
        50 mm markers on measuring tape A

    Cyan #00FFFF:
        50 mm markers on measuring tape B

    Green #00FF00:
        known 90 degree reference angle
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # OpenCV uses BGR instead of RGB.
    MAGENTA_BGR = (255, 0, 255)
    CYAN_BGR = (255, 255, 0)
    GREEN_BGR = (0, 255, 0)

    magenta_mask = _create_color_mask(
        image,
        MAGENTA_BGR,
        color_tolerance,
    )

    cyan_mask = _create_color_mask(
        image,
        CYAN_BGR,
        color_tolerance,
    )

    green_mask = _create_color_mask(
        image,
        GREEN_BGR,
        color_tolerance,
    )

    magenta_points = _find_dot_centers(
        magenta_mask,
        min_dot_area,
        max_dot_area,
    )

    cyan_points = _find_dot_centers(
        cyan_mask,
        min_dot_area,
        max_dot_area,
    )

    magenta_points = _sort_points_along_line(magenta_points)

    cyan_points = _sort_points_along_line(cyan_points)

    # np.where returns y, x
    green_y, green_x = np.where(green_mask > 0)

    green_pixels = np.column_stack((green_x, green_y)).astype(np.float64)

    return MarkerDetectionResult(
        magenta_points=magenta_points,
        cyan_points=cyan_points,
        green_pixels=green_pixels,
        magenta_mask=magenta_mask,
        cyan_mask=cyan_mask,
        green_mask=green_mask,
    )


def save_marker_debug_image(
    image_path: str,
    result: MarkerDetectionResult,
    output_path: str,
) -> None:
    """
    Save an image showing all detected marker centers.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Magenta points
    for i, (x, y) in enumerate(result.magenta_points):
        center = (round(x), round(y))

        cv2.circle(
            image,
            center,
            8,
            (255, 0, 255),
            2,
        )

        cv2.putText(
            image,
            f"M{i}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Cyan points
    for i, (x, y) in enumerate(result.cyan_points):
        center = (round(x), round(y))

        cv2.circle(
            image,
            center,
            8,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"C{i}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        image,
    )
