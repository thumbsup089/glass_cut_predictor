from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class AngleDetectionResult:
    corner: tuple[float, float]

    direction1: np.ndarray
    direction2: np.ndarray

    image_angle_deg: float

    line1_center: np.ndarray
    line2_center: np.ndarray


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def _line_length(line: np.ndarray) -> float:
    x1, y1, x2, y2 = line

    return float(
        np.hypot(
            x2 - x1,
            y2 - y1,
        )
    )


def _line_angle(line: np.ndarray) -> float:
    """
    Direction of an undirected line.

    0° and 180° are treated as identical.
    Result is between 0 and pi.
    """

    x1, y1, x2, y2 = line

    angle = np.arctan2(
        y2 - y1,
        x2 - x1,
    )

    return float(angle % np.pi)


def _angle_distance(
    angle1: float,
    angle2: float,
) -> float:
    """
    Smallest difference between two undirected lines.
    """

    diff = abs(angle1 - angle2)

    return float(
        min(
            diff,
            np.pi - diff,
        )
    )


def _mean_angle(
    angles: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted mean orientation.

    Because a line has no direction,
    theta and theta + 180° are identical.
    Therefore 2*theta is used.
    """

    x = np.sum(weights * np.cos(2.0 * angles))

    y = np.sum(weights * np.sin(2.0 * angles))

    angle = 0.5 * np.arctan2(y, x)

    return float(angle % np.pi)


# =========================================================
# CLUSTER HOUGH LINES INTO TWO ARMS
# =========================================================


def _cluster_lines(
    lines: np.ndarray,
):
    """
    Separate detected green lines into the
    two arms of the L-shaped marker.
    """

    angles = np.array(
        [_line_angle(line) for line in lines],
        dtype=np.float64,
    )

    lengths = np.array(
        [_line_length(line) for line in lines],
        dtype=np.float64,
    )

    # -----------------------------------------------------
    # First orientation:
    # use longest detected line
    # -----------------------------------------------------

    first_index = int(np.argmax(lengths))

    center1 = angles[first_index]

    # -----------------------------------------------------
    # Second orientation:
    # choose long line with largest angular difference
    # -----------------------------------------------------

    distances = np.array(
        [
            _angle_distance(
                angle,
                center1,
            )
            for angle in angles
        ]
    )

    score = distances * lengths

    second_index = int(np.argmax(score))

    center2 = angles[second_index]

    # -----------------------------------------------------
    # Iterative two-cluster fitting
    # -----------------------------------------------------

    for _ in range(50):

        distance1 = np.array(
            [
                _angle_distance(
                    angle,
                    center1,
                )
                for angle in angles
            ]
        )

        distance2 = np.array(
            [
                _angle_distance(
                    angle,
                    center2,
                )
                for angle in angles
            ]
        )

        cluster1 = distance1 <= distance2

        cluster2 = ~cluster1

        if not np.any(cluster1):
            raise RuntimeError("Could not detect green angle arm 1.")

        if not np.any(cluster2):
            raise RuntimeError("Could not detect green angle arm 2.")

        new_center1 = _mean_angle(
            angles[cluster1],
            lengths[cluster1],
        )

        new_center2 = _mean_angle(
            angles[cluster2],
            lengths[cluster2],
        )

        change = _angle_distance(
            center1,
            new_center1,
        ) + _angle_distance(
            center2,
            new_center2,
        )

        center1 = new_center1
        center2 = new_center2

        if change < 1e-8:
            break

    separation_deg = np.degrees(
        _angle_distance(
            center1,
            center2,
        )
    )

    if separation_deg < 20:
        raise RuntimeError(
            "The two green angle arms appear "
            "almost parallel. Angle detection failed."
        )

    return (
        lines[cluster1],
        lines[cluster2],
    )


# =========================================================
# FIT CENTER LINE THROUGH EACH THICK ARM
# =========================================================


def _fit_line_group(
    lines: np.ndarray,
):
    """
    Fit one center line through all Hough segments
    belonging to one arm.

    This averages the two edges of the thick
    green painted line.
    """

    points = []

    for line in lines:

        x1, y1, x2, y2 = line

        points.append([x1, y1])

        points.append([x2, y2])

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).flatten()

    direction = np.array(
        [vx, vy],
        dtype=np.float64,
    )

    direction /= np.linalg.norm(direction)

    center = np.array(
        [x0, y0],
        dtype=np.float64,
    )

    return (
        center,
        direction,
    )


# =========================================================
# LINE INTERSECTION
# =========================================================


def _line_intersection(
    center1: np.ndarray,
    direction1: np.ndarray,
    center2: np.ndarray,
    direction2: np.ndarray,
) -> np.ndarray:

    matrix = np.column_stack(
        (
            direction1,
            -direction2,
        )
    )

    determinant = np.linalg.det(matrix)

    if abs(determinant) < 1e-10:

        raise RuntimeError("Detected green angle lines are parallel.")

    rhs = center2 - center1

    t, _ = np.linalg.solve(
        matrix,
        rhs,
    )

    corner = center1 + t * direction1

    return corner


# =========================================================
# ANGLE
# =========================================================


def _angle_between_lines(
    direction1: np.ndarray,
    direction2: np.ndarray,
) -> float:

    dot = abs(
        np.dot(
            direction1,
            direction2,
        )
    )

    dot = np.clip(
        dot,
        -1.0,
        1.0,
    )

    angle_rad = np.arccos(dot)

    return float(np.degrees(angle_rad))


# =========================================================
# MAIN DETECTION FUNCTION
# =========================================================


def detect_green_angle(
    green_mask: np.ndarray,
    hough_threshold: int = 40,
    min_line_length: int = 80,
    max_line_gap: int = 20,
) -> AngleDetectionResult:
    """
    Detect the two arms of the green L-shaped
    90-degree calibration reference.

    IMPORTANT:
    The function does NOT assume that the angle
    already appears as 90° in the photograph.

    It measures the angle as visible in the image.
    """

    if green_mask is None:

        raise ValueError("green_mask is None.")

    green_pixel_count = int(np.count_nonzero(green_mask))

    if green_pixel_count < 100:

        raise RuntimeError("Not enough green pixels detected.")

    # -----------------------------------------------------
    # Close small holes / gaps
    # -----------------------------------------------------

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    cleaned_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # -----------------------------------------------------
    # Find straight pieces of the green L
    # -----------------------------------------------------

    detected = cv2.HoughLinesP(
        cleaned_mask,
        rho=1,
        theta=np.pi / 360.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if detected is None:

        raise RuntimeError("No straight green lines detected.")

    lines = np.asarray(
        detected,
        dtype=np.float64,
    ).reshape(-1, 4)

    if len(lines) < 2:

        raise RuntimeError("Not enough green lines detected.")

    print()
    print(f"Green Hough lines detected: " f"{len(lines)}")

    # -----------------------------------------------------
    # Separate horizontal-ish and vertical-ish arm
    # WITHOUT assuming their actual orientation.
    # -----------------------------------------------------

    arm1_lines, arm2_lines = _cluster_lines(lines)

    print(f"Angle arm 1 lines: " f"{len(arm1_lines)}")

    print(f"Angle arm 2 lines: " f"{len(arm2_lines)}")

    # -----------------------------------------------------
    # Fit one center line through each thick arm
    # -----------------------------------------------------

    line1_center, direction1 = _fit_line_group(arm1_lines)

    line2_center, direction2 = _fit_line_group(arm2_lines)

    # -----------------------------------------------------
    # Find actual L corner
    # -----------------------------------------------------

    corner = _line_intersection(
        line1_center,
        direction1,
        line2_center,
        direction2,
    )

    # -----------------------------------------------------
    # Angle as visible in photograph
    # -----------------------------------------------------

    image_angle_deg = _angle_between_lines(
        direction1,
        direction2,
    )

    return AngleDetectionResult(
        corner=(
            float(corner[0]),
            float(corner[1]),
        ),
        direction1=direction1,
        direction2=direction2,
        image_angle_deg=image_angle_deg,
        line1_center=line1_center,
        line2_center=line2_center,
    )


# =========================================================
# DEBUG IMAGE
# =========================================================


def save_angle_debug_image(
    image_path: str,
    result: AngleDetectionResult,
    output_path: str,
) -> None:

    image = cv2.imread(image_path)

    if image is None:

        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]

    extension = (
        max(
            width,
            height,
        )
        * 2
    )

    # -----------------------------------------------------
    # LINE 1 - RED
    # -----------------------------------------------------

    start1 = result.line1_center - result.direction1 * extension

    end1 = result.line1_center + result.direction1 * extension

    cv2.line(
        image,
        tuple(np.round(start1).astype(int)),
        tuple(np.round(end1).astype(int)),
        (0, 0, 255),
        3,
    )

    # -----------------------------------------------------
    # LINE 2 - BLUE
    # -----------------------------------------------------

    start2 = result.line2_center - result.direction2 * extension

    end2 = result.line2_center + result.direction2 * extension

    cv2.line(
        image,
        tuple(np.round(start2).astype(int)),
        tuple(np.round(end2).astype(int)),
        (255, 0, 0),
        3,
    )

    # -----------------------------------------------------
    # CORNER - YELLOW
    # -----------------------------------------------------

    corner = tuple(np.round(result.corner).astype(int))

    cv2.circle(
        image,
        corner,
        10,
        (0, 255, 255),
        -1,
    )

    # -----------------------------------------------------
    # TEXT
    # -----------------------------------------------------

    cv2.putText(
        image,
        (f"Angle in photo: " f"{result.image_angle_deg:.2f} deg"),
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
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
