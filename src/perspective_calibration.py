from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

# =========================================================
# RESULT
# =========================================================


@dataclass
class PerspectiveCalibrationResult:
    # Homography:
    # input image pixels -> real world millimeters
    pixel_to_mm: np.ndarray

    # Homography in normalized image coordinates
    normalized_homography: np.ndarray

    # Validation
    magenta_distances_mm: list[float]
    cyan_distances_mm: list[float]

    mean_spacing_error_mm: float
    max_spacing_error_mm: float

    green_angle_deg: float


# =========================================================
# BASIC HOMOGRAPHY FUNCTIONS
# =========================================================


def _build_homography(
    params: np.ndarray,
) -> np.ndarray:
    """
    Build 3x3 homography from 8 parameters.

    Last element is fixed to 1 because homographies
    are only defined up to scale.
    """

    return np.array(
        [
            [
                params[0],
                params[1],
                params[2],
            ],
            [
                params[3],
                params[4],
                params[5],
            ],
            [
                params[6],
                params[7],
                1.0,
            ],
        ],
        dtype=np.float64,
    )


def _homography_to_params(
    H: np.ndarray,
) -> np.ndarray:

    H = H / H[2, 2]

    return np.array(
        [
            H[0, 0],
            H[0, 1],
            H[0, 2],
            H[1, 0],
            H[1, 1],
            H[1, 2],
            H[2, 0],
            H[2, 1],
        ],
        dtype=np.float64,
    )


# =========================================================
# NORMALIZED IMAGE COORDINATES
# =========================================================


def _pixel_to_normalized_matrix(
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert pixel coordinates to approximately
    -0.5 .. +0.5 normalized image coordinates.

    Normalization makes numerical optimization much
    more stable than directly optimizing ~2000 px values.
    """

    return np.array(
        [
            [
                1.0 / width,
                0.0,
                -0.5,
            ],
            [
                0.0,
                1.0 / height,
                -0.5,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )


# =========================================================
# POINT TRANSFORMATION
# =========================================================


def _transform_points(
    points: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    ones = np.ones(
        (
            len(points),
            1,
        ),
        dtype=np.float64,
    )

    homogeneous = np.hstack(
        (
            points,
            ones,
        )
    )

    transformed = (H @ homogeneous.T).T

    denominator = transformed[:, 2]

    if np.any(np.abs(denominator) < 1e-10):
        raise ValueError("Homography produced invalid coordinates.")

    transformed = transformed[:, :2] / denominator[:, None]

    return transformed


def transform_image_points_to_mm(
    points: list[tuple[float, float]],
    calibration: PerspectiveCalibrationResult,
) -> np.ndarray:
    """
    Public helper:
    image pixel coordinates -> millimeter coordinates.
    """

    return _transform_points(
        np.asarray(
            points,
            dtype=np.float64,
        ),
        calibration.pixel_to_mm,
    )


# =========================================================
# GREEN AXIS SELECTION
# =========================================================


def _choose_green_axes(
    green_angle,
):
    """
    Decide which detected green arm should become
    horizontal and which should become vertical.

    Direction signs are chosen so that:

    horizontal -> roughly image-right
    vertical   -> roughly image-down
    """

    d1 = np.asarray(
        green_angle.direction1,
        dtype=np.float64,
    )

    d2 = np.asarray(
        green_angle.direction2,
        dtype=np.float64,
    )

    c1 = np.asarray(
        green_angle.line1_center,
        dtype=np.float64,
    )

    c2 = np.asarray(
        green_angle.line2_center,
        dtype=np.float64,
    )

    horizontal_score_1 = abs(d1[0]) / (abs(d1[0]) + abs(d1[1]) + 1e-12)

    horizontal_score_2 = abs(d2[0]) / (abs(d2[0]) + abs(d2[1]) + 1e-12)

    if horizontal_score_1 >= horizontal_score_2:

        horizontal_direction = d1.copy()
        vertical_direction = d2.copy()

        horizontal_center = c1.copy()
        vertical_center = c2.copy()

    else:

        horizontal_direction = d2.copy()
        vertical_direction = d1.copy()

        horizontal_center = c2.copy()
        vertical_center = c1.copy()

    # Right should stay right.
    if horizontal_direction[0] < 0:
        horizontal_direction *= -1

    # Down should stay down.
    if vertical_direction[1] < 0:
        vertical_direction *= -1

    return (
        horizontal_center,
        horizontal_direction,
        vertical_center,
        vertical_direction,
    )


# =========================================================
# INITIAL HOMOGRAPHY
# =========================================================


def _initial_homography(
    width: int,
    height: int,
    magenta_points,
    cyan_points,
    green_angle,
    spacing_mm: float,
) -> np.ndarray:
    """
    Construct a good affine starting point.

    The green arms are initially mapped to X/Y axes,
    then the approximate scale is derived from the
    50 mm marker distances.
    """

    (
        _,
        horizontal_direction,
        _,
        vertical_direction,
    ) = _choose_green_axes(green_angle)

    # -----------------------------------------------------
    # Directions in normalized coordinates
    # -----------------------------------------------------

    dh = np.array(
        [
            horizontal_direction[0] / width,
            horizontal_direction[1] / height,
        ],
        dtype=np.float64,
    )

    dv = np.array(
        [
            vertical_direction[0] / width,
            vertical_direction[1] / height,
        ],
        dtype=np.float64,
    )

    basis = np.column_stack(
        (
            dh,
            dv,
        )
    )

    if abs(np.linalg.det(basis)) < 1e-10:
        raise RuntimeError("Green angle directions are nearly parallel.")

    A = np.linalg.inv(basis)

    # -----------------------------------------------------
    # Green corner becomes (0, 0)
    # -----------------------------------------------------

    corner = np.asarray(
        green_angle.corner,
        dtype=np.float64,
    )

    corner_normalized = np.array(
        [
            corner[0] / width - 0.5,
            corner[1] / height - 0.5,
        ],
        dtype=np.float64,
    )

    translation = -A @ corner_normalized

    H = np.array(
        [
            [
                A[0, 0],
                A[0, 1],
                translation[0],
            ],
            [
                A[1, 0],
                A[1, 1],
                translation[1],
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    # -----------------------------------------------------
    # Determine approximate metric scale
    # -----------------------------------------------------

    N = _pixel_to_normalized_matrix(
        width,
        height,
    )

    H_pixel = H @ N

    distances = []

    for points in (
        magenta_points,
        cyan_points,
    ):

        if len(points) < 2:
            continue

        transformed = _transform_points(
            np.asarray(
                points,
                dtype=np.float64,
            ),
            H_pixel,
        )

        diffs = np.diff(
            transformed,
            axis=0,
        )

        distances.extend(
            np.linalg.norm(
                diffs,
                axis=1,
            )
        )

    if not distances:
        raise RuntimeError("Not enough tape markers for scale estimation.")

    mean_distance = float(np.mean(distances))

    scale = spacing_mm / mean_distance

    H[0, :] *= scale
    H[1, :] *= scale

    return H


# =========================================================
# TAPE RESIDUALS
# =========================================================


def _tape_residuals(
    points_mm: np.ndarray,
    spacing_mm: float,
) -> list[float]:
    """
    Enforce:

    1. every neighboring marker = spacing_mm
    2. the points form one evenly spaced straight line
    """

    residuals = []

    if len(points_mm) < 2:
        return residuals

    differences = np.diff(
        points_mm,
        axis=0,
    )

    lengths = np.linalg.norm(
        differences,
        axis=1,
    )

    # -----------------------------------------------------
    # Every step must be 50 mm
    # -----------------------------------------------------

    for length in lengths:

        residuals.append((length - spacing_mm) / spacing_mm)

    # -----------------------------------------------------
    # Equal vector steps:
    #
    # P(i+1) - 2P(i) + P(i-1) = 0
    #
    # This simultaneously enforces straightness
    # and equal spacing direction.
    # -----------------------------------------------------

    if len(points_mm) >= 3:

        second_difference = points_mm[2:] - 2.0 * points_mm[1:-1] + points_mm[:-2]

        second_difference /= spacing_mm

        for dx, dy in second_difference:

            residuals.append(0.75 * dx)

            residuals.append(0.75 * dy)

    return residuals


# =========================================================
# CALIBRATION
# =========================================================


def calibrate_perspective(
    image_shape,
    magenta_points,
    cyan_points,
    green_angle,
    marker_spacing_mm: float = 50.0,
) -> PerspectiveCalibrationResult:
    """
    Estimate a projective transformation that maps
    the photographed plane into real metric coordinates.

    Constraints:

    - Magenta markers have 50 mm spacing.
    - Cyan markers have 50 mm spacing.
    - Both measuring tapes become straight.
    - Green reference arm 1 becomes horizontal.
    - Green reference arm 2 becomes vertical.
    - Therefore the green reference becomes exactly 90°.
    """

    height, width = image_shape[:2]

    magenta = np.asarray(
        magenta_points,
        dtype=np.float64,
    )

    cyan = np.asarray(
        cyan_points,
        dtype=np.float64,
    )

    N = _pixel_to_normalized_matrix(
        width,
        height,
    )

    # -----------------------------------------------------
    # Initial solution
    # -----------------------------------------------------

    H_initial = _initial_homography(
        width,
        height,
        magenta_points,
        cyan_points,
        green_angle,
        marker_spacing_mm,
    )

    initial_params = _homography_to_params(H_initial)

    (
        horizontal_center,
        horizontal_direction,
        vertical_center,
        vertical_direction,
    ) = _choose_green_axes(green_angle)

    corner = np.asarray(
        green_angle.corner,
        dtype=np.float64,
    )

    # Sample points along the two green lines.
    # The actual length is irrelevant because a homography
    # maps the whole line.
    sample_length = 0.20 * max(
        width,
        height,
    )

    horizontal_samples = np.array(
        [
            horizontal_center - horizontal_direction * sample_length,
            horizontal_center + horizontal_direction * sample_length,
        ],
        dtype=np.float64,
    )

    vertical_samples = np.array(
        [
            vertical_center - vertical_direction * sample_length,
            vertical_center + vertical_direction * sample_length,
        ],
        dtype=np.float64,
    )

    # -----------------------------------------------------
    # Optimization residual
    # -----------------------------------------------------

    def residual_function(
        params: np.ndarray,
    ) -> np.ndarray:

        H_normalized = _build_homography(params)

        H_pixel = H_normalized @ N

        try:

            magenta_mm = _transform_points(
                magenta,
                H_pixel,
            )

            cyan_mm = _transform_points(
                cyan,
                H_pixel,
            )

            corner_mm = _transform_points(
                np.array([corner]),
                H_pixel,
            )[0]

            horizontal_mm = _transform_points(
                horizontal_samples,
                H_pixel,
            )

            vertical_mm = _transform_points(
                vertical_samples,
                H_pixel,
            )

        except Exception:

            return np.full(
                64,
                1e6,
                dtype=np.float64,
            )

        residuals = []

        # ---------------------------------------------
        # Measuring tapes
        # ---------------------------------------------

        residuals.extend(
            _tape_residuals(
                magenta_mm,
                marker_spacing_mm,
            )
        )

        residuals.extend(
            _tape_residuals(
                cyan_mm,
                marker_spacing_mm,
            )
        )

        # ---------------------------------------------
        # Green corner must become origin
        # ---------------------------------------------

        green_weight = 10.0

        residuals.append(green_weight * corner_mm[0] / marker_spacing_mm)

        residuals.append(green_weight * corner_mm[1] / marker_spacing_mm)

        # ---------------------------------------------
        # Horizontal green arm -> y = 0
        # ---------------------------------------------

        for point in horizontal_mm:

            residuals.append(green_weight * point[1] / marker_spacing_mm)

        # ---------------------------------------------
        # Vertical green arm -> x = 0
        # ---------------------------------------------

        for point in vertical_mm:

            residuals.append(green_weight * point[0] / marker_spacing_mm)

        # ---------------------------------------------
        # Very weak projective regularization.
        #
        # Prevents absurd solutions when annotations
        # contain noise.
        # ---------------------------------------------

        residuals.append(0.001 * params[6])

        residuals.append(0.001 * params[7])

        return np.asarray(
            residuals,
            dtype=np.float64,
        )

    # -----------------------------------------------------
    # Optimize
    # -----------------------------------------------------

    optimization = least_squares(
        residual_function,
        initial_params,
        method="trf",
        loss="soft_l1",
        max_nfev=5000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    if not optimization.success:

        raise RuntimeError("Perspective optimization failed: " + optimization.message)

    H_normalized = _build_homography(optimization.x)

    H_pixel_to_mm = H_normalized @ N

    # Normalize matrix.
    H_pixel_to_mm /= H_pixel_to_mm[2, 2]

    # =====================================================
    # VALIDATION
    # =====================================================

    magenta_mm = _transform_points(
        magenta,
        H_pixel_to_mm,
    )

    cyan_mm = _transform_points(
        cyan,
        H_pixel_to_mm,
    )

    magenta_distances = np.linalg.norm(
        np.diff(
            magenta_mm,
            axis=0,
        ),
        axis=1,
    )

    cyan_distances = np.linalg.norm(
        np.diff(
            cyan_mm,
            axis=0,
        ),
        axis=1,
    )

    all_distances = np.concatenate(
        (
            magenta_distances,
            cyan_distances,
        )
    )

    errors = np.abs(all_distances - marker_spacing_mm)

    # -----------------------------------------------------
    # Green angle validation
    # -----------------------------------------------------

    horizontal_mm = _transform_points(
        horizontal_samples,
        H_pixel_to_mm,
    )

    vertical_mm = _transform_points(
        vertical_samples,
        H_pixel_to_mm,
    )

    horizontal_vector = horizontal_mm[1] - horizontal_mm[0]

    vertical_vector = vertical_mm[1] - vertical_mm[0]

    horizontal_vector /= np.linalg.norm(horizontal_vector)

    vertical_vector /= np.linalg.norm(vertical_vector)

    dot = abs(
        np.dot(
            horizontal_vector,
            vertical_vector,
        )
    )

    dot = np.clip(
        dot,
        -1.0,
        1.0,
    )

    green_angle_deg = float(np.degrees(np.arccos(dot)))

    return PerspectiveCalibrationResult(
        pixel_to_mm=H_pixel_to_mm,
        normalized_homography=H_normalized,
        magenta_distances_mm=[float(value) for value in magenta_distances],
        cyan_distances_mm=[float(value) for value in cyan_distances],
        mean_spacing_error_mm=float(np.mean(errors)),
        max_spacing_error_mm=float(np.max(errors)),
        green_angle_deg=green_angle_deg,
    )


# =========================================================
# RECTIFY COMPLETE IMAGE
# =========================================================


def rectify_image(
    image: np.ndarray,
    calibration: PerspectiveCalibrationResult,
    pixels_per_mm: float = 4.0,
    padding_mm: float = 10.0,
) -> np.ndarray:
    """
    Transform the entire source image into an orthogonal,
    metric top-down view.

    Output resolution:
        pixels_per_mm pixels = 1 mm
    """

    height, width = image.shape[:2]

    image_corners = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float64,
    )

    world_corners = _transform_points(
        image_corners,
        calibration.pixel_to_mm,
    )

    min_x = float(np.min(world_corners[:, 0]) - padding_mm)

    max_x = float(np.max(world_corners[:, 0]) + padding_mm)

    min_y = float(np.min(world_corners[:, 1]) - padding_mm)

    max_y = float(np.max(world_corners[:, 1]) + padding_mm)

    output_width = int(np.ceil((max_x - min_x) * pixels_per_mm))

    output_height = int(np.ceil((max_y - min_y) * pixels_per_mm))

    if output_width <= 0 or output_height <= 0:
        raise RuntimeError("Invalid rectified image size.")

    if output_width > 20000 or output_height > 20000:
        raise RuntimeError(
            "Rectified image became unreasonably large. "
            "Calibration is probably incorrect."
        )

    # -----------------------------------------------------
    # Millimeters -> output pixels
    # -----------------------------------------------------

    mm_to_output = np.array(
        [
            [
                pixels_per_mm,
                0.0,
                -min_x * pixels_per_mm,
            ],
            [
                0.0,
                pixels_per_mm,
                -min_y * pixels_per_mm,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    pixel_to_output = mm_to_output @ calibration.pixel_to_mm

    rectified = cv2.warpPerspective(
        image,
        pixel_to_output,
        (
            output_width,
            output_height,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            255,
            255,
            255,
        ),
    )

    return rectified


# =========================================================
# PRINT VALIDATION REPORT
# =========================================================


def print_perspective_report(
    calibration: PerspectiveCalibrationResult,
):

    print()
    print("========================================")
    print("PERSPECTIVE CALIBRATION")
    print("========================================")

    print()
    print("MAGENTA:")

    for i, distance in enumerate(calibration.magenta_distances_mm):

        print(f"M{i} -> M{i + 1}: " f"{distance:.3f} mm")

    print()
    print("CYAN:")

    for i, distance in enumerate(calibration.cyan_distances_mm):

        print(f"C{i} -> C{i + 1}: " f"{distance:.3f} mm")

    print()
    print(f"Green angle: " f"{calibration.green_angle_deg:.4f}°")

    print(f"Mean spacing error: " f"{calibration.mean_spacing_error_mm:.4f} mm")

    print(f"Maximum spacing error: " f"{calibration.max_spacing_error_mm:.4f} mm")
