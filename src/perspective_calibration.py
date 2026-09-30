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

    # Distances of ALL detected neighboring marker segments.
    # Excluded outlier segments are still listed here for debugging.
    magenta_distances_mm: list[float]
    cyan_distances_mm: list[float]

    # Validation error calculated ONLY from retained/valid segments.
    mean_spacing_error_mm: float
    max_spacing_error_mm: float

    green_angle_deg: float

    # Example: ["C8-C9", "M0-M1"]
    excluded_segments: list[str]


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
    marker distances.
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

    # Median is less sensitive to a single bad marker
    # than the mean.
    median_distance = float(np.median(distances))

    scale = spacing_mm / median_distance

    H[0, :] *= scale
    H[1, :] *= scale

    return H


# =========================================================
# TAPE RESIDUALS
# =========================================================


def _tape_residuals(
    points_mm: np.ndarray,
    spacing_mm: float,
    valid_segments: np.ndarray,
) -> list[float]:
    """
    Enforce for VALID segments only:

    1. every neighboring marker = spacing_mm
    2. neighboring valid segments form one evenly
       spaced straight line

    An excluded segment is ignored by the optimizer.
    """

    residuals = []

    if len(points_mm) < 2:
        return residuals

    valid_segments = np.asarray(
        valid_segments,
        dtype=bool,
    )

    if len(valid_segments) != len(points_mm) - 1:
        raise ValueError("valid_segments has incorrect length.")

    differences = np.diff(
        points_mm,
        axis=0,
    )

    lengths = np.linalg.norm(
        differences,
        axis=1,
    )

    # -----------------------------------------------------
    # Every VALID step must be spacing_mm
    # -----------------------------------------------------

    for index, length in enumerate(lengths):

        if not valid_segments[index]:
            continue

        residuals.append((length - spacing_mm) / spacing_mm)

    # -----------------------------------------------------
    # Equal vector steps / straightness.
    #
    # Only use a middle marker if BOTH adjacent
    # intervals are still valid.
    # -----------------------------------------------------

    for index in range(
        1,
        len(points_mm) - 1,
    ):

        if not (valid_segments[index - 1] and valid_segments[index]):
            continue

        second_difference = (
            points_mm[index + 1] - 2.0 * points_mm[index] + points_mm[index - 1]
        )

        second_difference /= spacing_mm

        residuals.append(0.75 * second_difference[0])

        residuals.append(0.75 * second_difference[1])

    return residuals


def _tape_residual_count(
    valid_segments: np.ndarray,
) -> int:
    """
    Number of residuals produced by _tape_residuals().
    Used so the error fallback inside least_squares
    always has the correct vector length.
    """

    valid_segments = np.asarray(
        valid_segments,
        dtype=bool,
    )

    distance_count = int(np.count_nonzero(valid_segments))

    straightness_count = 0

    if len(valid_segments) >= 2:

        straightness_count = 2 * int(
            np.count_nonzero(valid_segments[:-1] & valid_segments[1:])
        )

    return distance_count + straightness_count


# =========================================================
# OUTLIER DETECTION
# =========================================================


def _find_clear_outlier_candidate(
    magenta_distances_mm: np.ndarray,
    cyan_distances_mm: np.ndarray,
    magenta_valid: np.ndarray,
    cyan_valid: np.ndarray,
    target_mm: float,
    minimum_error_mm: float = 2.0,
    mad_factor: float = 4.0,
):
    """
    Find one CLEAR outlier candidate across both measuring tapes.

    A segment is only considered a candidate if its absolute error is
    larger than BOTH:

    1. minimum_error_mm
    2. a data-dependent robust threshold based on Median + MAD

    The candidate is NOT removed here. It still has to pass the
    leave-one-out improvement test inside calibrate_perspective().
    """

    magenta_errors = np.abs(
        np.asarray(
            magenta_distances_mm,
            dtype=np.float64,
        )
        - target_mm
    )

    cyan_errors = np.abs(
        np.asarray(
            cyan_distances_mm,
            dtype=np.float64,
        )
        - target_mm
    )

    active_errors = np.concatenate(
        (
            magenta_errors[
                np.asarray(
                    magenta_valid,
                    dtype=bool,
                )
            ],
            cyan_errors[
                np.asarray(
                    cyan_valid,
                    dtype=bool,
                )
            ],
        )
    )

    if len(active_errors) == 0:
        return None

    median_error = float(np.median(active_errors))

    mad = float(np.median(np.abs(active_errors - median_error)))

    robust_threshold = median_error + mad_factor * 1.4826 * mad

    threshold_mm = max(
        float(minimum_error_mm),
        float(robust_threshold),
    )

    candidates = []

    for index, error in enumerate(magenta_errors):

        if magenta_valid[index] and error > threshold_mm:

            candidates.append(
                (
                    float(error),
                    "magenta",
                    int(index),
                )
            )

    for index, error in enumerate(cyan_errors):

        if cyan_valid[index] and error > threshold_mm:

            candidates.append(
                (
                    float(error),
                    "cyan",
                    int(index),
                )
            )

    if not candidates:
        return None

    error_mm, tape_name, segment_index = max(
        candidates,
        key=lambda item: item[0],
    )

    return {
        "tape": tape_name,
        "index": segment_index,
        "error_mm": error_mm,
        "threshold_mm": threshold_mm,
        "median_error_mm": median_error,
        "mad_mm": mad,
    }


def _mean_error_for_masks(
    magenta_distances_mm: np.ndarray,
    cyan_distances_mm: np.ndarray,
    magenta_valid: np.ndarray,
    cyan_valid: np.ndarray,
    target_mm: float,
) -> float:
    """
    Mean absolute spacing error for exactly the supplied masks.

    This is important for the leave-one-out test:
    before and after recalibration are compared on the SAME
    retained segments, so the improvement cannot come merely
    from deleting one bad value from the average.
    """

    errors = []

    magenta_distances_mm = np.asarray(
        magenta_distances_mm,
        dtype=np.float64,
    )

    cyan_distances_mm = np.asarray(
        cyan_distances_mm,
        dtype=np.float64,
    )

    for distance, valid in zip(
        magenta_distances_mm,
        magenta_valid,
    ):

        if valid:
            errors.append(abs(float(distance) - target_mm))

    for distance, valid in zip(
        cyan_distances_mm,
        cyan_valid,
    ):

        if valid:
            errors.append(abs(float(distance) - target_mm))

    if not errors:
        raise RuntimeError("No calibration segments available for error comparison.")

    return float(np.mean(errors))


# =========================================================
# CALIBRATION
# =========================================================


def calibrate_perspective(
    image_shape,
    magenta_points,
    cyan_points,
    green_angle,
    marker_spacing_mm: float = 50.0,
    min_valid_segments_per_tape: int = 3,
    outlier_minimum_error_mm: float = 2.0,
    outlier_mad_factor: float = 4.0,
    outlier_min_improvement_ratio: float = 0.50,
    max_outlier_exclusions: int = 8,
) -> PerspectiveCalibrationResult:
    """
    Estimate a projective transformation that maps
    the photographed plane into real metric coordinates.

    Constraints:

    - Magenta markers have marker_spacing_mm spacing.
    - Cyan markers have marker_spacing_mm spacing.
    - Both measuring tapes become straight.
    - Green reference arm 1 becomes horizontal.
    - Green reference arm 2 becomes vertical.
    - Therefore the green reference becomes exactly 90°.

    Robust outlier logic:

    1. Calibrate with all currently valid intervals.
    2. Search for a CLEAR candidate:
       error > max(
           outlier_minimum_error_mm,
           Median + MAD based threshold
       )
    3. Temporarily exclude ONLY that one interval.
    4. Recalibrate.
    5. Compare old vs new calibration on the SAME retained
       intervals.
    6. Permanently exclude the segment only if the mean error
       improves by at least outlier_min_improvement_ratio.

    This prevents ordinary ~0.3...0.7 mm measurement noise
    from being removed just because it is the worst value in
    an otherwise good image.

    If a large inconsistency exists but removing the suspected
    segment does not clearly improve the calibration, the image
    is rejected for manual inspection instead of silently
    deleting data.
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

    magenta_segment_count = len(magenta) - 1

    cyan_segment_count = len(cyan) - 1

    if magenta_segment_count < min_valid_segments_per_tape:

        raise RuntimeError(
            "Not enough magenta calibration data. "
            f"Need at least "
            f"{min_valid_segments_per_tape + 1} markers "
            f"({min_valid_segments_per_tape} intervals), "
            f"but found {len(magenta)} markers."
        )

    if cyan_segment_count < min_valid_segments_per_tape:

        raise RuntimeError(
            "Not enough cyan calibration data. "
            f"Need at least "
            f"{min_valid_segments_per_tape + 1} markers "
            f"({min_valid_segments_per_tape} intervals), "
            f"but found {len(cyan)} markers."
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

    current_params = _homography_to_params(H_initial)

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
    # Initially every interval is trusted.
    # -----------------------------------------------------

    magenta_valid = np.ones(
        magenta_segment_count,
        dtype=bool,
    )

    cyan_valid = np.ones(
        cyan_segment_count,
        dtype=bool,
    )

    excluded_segments = []

    # -----------------------------------------------------
    # Optimization helper
    # -----------------------------------------------------

    def optimize_with_masks(
        start_params: np.ndarray,
        magenta_mask: np.ndarray,
        cyan_mask: np.ndarray,
    ):
        """
        Run one full perspective optimization for the supplied
        valid/invalid segment masks.
        """

        def residual_function(
            params: np.ndarray,
        ) -> np.ndarray:

            H_iteration = _build_homography(params)

            H_pixel = H_iteration @ N

            expected_residual_count = (
                _tape_residual_count(magenta_mask) + _tape_residual_count(cyan_mask) + 8
            )

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
                    expected_residual_count,
                    1e6,
                    dtype=np.float64,
                )

            residuals = []

            residuals.extend(
                _tape_residuals(
                    magenta_mm,
                    marker_spacing_mm,
                    magenta_mask,
                )
            )

            residuals.extend(
                _tape_residuals(
                    cyan_mm,
                    marker_spacing_mm,
                    cyan_mask,
                )
            )

            green_weight = 10.0

            # Green corner -> origin
            residuals.append(green_weight * corner_mm[0] / marker_spacing_mm)

            residuals.append(green_weight * corner_mm[1] / marker_spacing_mm)

            # Horizontal green arm -> y = 0
            for point in horizontal_mm:

                residuals.append(green_weight * point[1] / marker_spacing_mm)

            # Vertical green arm -> x = 0
            for point in vertical_mm:

                residuals.append(green_weight * point[0] / marker_spacing_mm)

            # Weak projective regularization
            residuals.append(0.001 * params[6])

            residuals.append(0.001 * params[7])

            return np.asarray(
                residuals,
                dtype=np.float64,
            )

        optimization = least_squares(
            residual_function,
            start_params,
            method="trf",
            loss="soft_l1",
            max_nfev=5000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )

        if not optimization.success:

            raise RuntimeError(
                "Perspective optimization failed: " + optimization.message
            )

        H_normalized = _build_homography(optimization.x)

        H_pixel_to_mm = H_normalized @ N

        H_pixel_to_mm /= H_pixel_to_mm[
            2,
            2,
        ]

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

        return {
            "params": optimization.x,
            "H_normalized": H_normalized,
            "H_pixel_to_mm": H_pixel_to_mm,
            "magenta_distances": magenta_distances,
            "cyan_distances": cyan_distances,
        }

    # =====================================================
    # INITIAL CALIBRATION
    # =====================================================

    current = optimize_with_masks(
        current_params,
        magenta_valid,
        cyan_valid,
    )

    # =====================================================
    # LEAVE-ONE-OUT OUTLIER CONFIRMATION
    # =====================================================

    exclusion_count = 0

    while True:

        candidate = _find_clear_outlier_candidate(
            current["magenta_distances"],
            current["cyan_distances"],
            magenta_valid,
            cyan_valid,
            marker_spacing_mm,
            minimum_error_mm=(outlier_minimum_error_mm),
            mad_factor=(outlier_mad_factor),
        )

        # No large + statistically unusual error remains.
        if candidate is None:
            break

        if exclusion_count >= max_outlier_exclusions:

            raise RuntimeError(
                "Too many calibration outliers. " "Image should be checked manually."
            )

        tape_name = candidate["tape"]

        segment_index = candidate["index"]

        error_mm = candidate["error_mm"]

        threshold_mm = candidate["threshold_mm"]

        if tape_name == "magenta":

            label = f"M{segment_index}" f"-M{segment_index + 1}"

            test_magenta_valid = magenta_valid.copy()

            test_cyan_valid = cyan_valid.copy()

            test_magenta_valid[segment_index] = False

            remaining = int(np.count_nonzero(test_magenta_valid))

            if remaining < min_valid_segments_per_tape:

                raise RuntimeError(
                    "Calibration rejected: "
                    f"{label} is a strong outlier candidate "
                    f"(error {error_mm:.3f} mm), "
                    "but excluding it would leave only "
                    f"{remaining} reliable magenta intervals. "
                    f"Need at least "
                    f"{min_valid_segments_per_tape}."
                )

        else:

            label = f"C{segment_index}" f"-C{segment_index + 1}"

            test_magenta_valid = magenta_valid.copy()

            test_cyan_valid = cyan_valid.copy()

            test_cyan_valid[segment_index] = False

            remaining = int(np.count_nonzero(test_cyan_valid))

            if remaining < min_valid_segments_per_tape:

                raise RuntimeError(
                    "Calibration rejected: "
                    f"{label} is a strong outlier candidate "
                    f"(error {error_mm:.3f} mm), "
                    "but excluding it would leave only "
                    f"{remaining} reliable cyan intervals. "
                    f"Need at least "
                    f"{min_valid_segments_per_tape}."
                )

        # -------------------------------------------------
        # Compare old vs new fit on the SAME retained data.
        # -------------------------------------------------

        before_common_error = _mean_error_for_masks(
            current["magenta_distances"],
            current["cyan_distances"],
            test_magenta_valid,
            test_cyan_valid,
            marker_spacing_mm,
        )

        test_result = optimize_with_masks(
            current["params"],
            test_magenta_valid,
            test_cyan_valid,
        )

        after_common_error = _mean_error_for_masks(
            test_result["magenta_distances"],
            test_result["cyan_distances"],
            test_magenta_valid,
            test_cyan_valid,
            marker_spacing_mm,
        )

        if before_common_error <= 1e-12:

            improvement_ratio = 0.0

        else:

            improvement_ratio = (
                before_common_error - after_common_error
            ) / before_common_error

        print(
            f"Outlier candidate: "
            f"{label}"
            f" | error={error_mm:.3f} mm"
            f" | threshold={threshold_mm:.3f} mm"
        )

        print(
            f"Leave-one-out test: "
            f"{before_common_error:.3f} mm"
            f" -> {after_common_error:.3f} mm"
            f" | improvement="
            f"{improvement_ratio * 100.0:.1f}%"
        )

        # -------------------------------------------------
        # Only NOW is the candidate really excluded.
        # -------------------------------------------------

        if improvement_ratio >= outlier_min_improvement_ratio:

            magenta_valid = test_magenta_valid

            cyan_valid = test_cyan_valid

            current = test_result

            excluded_segments.append(label)

            exclusion_count += 1

            print(f"Confirmed outlier: " f"{label} excluded.")

            print("Recalibrating / checking again...")

            continue

        # A large inconsistency exists, but it cannot be
        # explained cleanly by this one segment.
        raise RuntimeError(
            "Calibration contains a large inconsistency, "
            f"but {label} was not confirmed as a clear "
            "single outlier. "
            f"Leave-one-out improvement was only "
            f"{improvement_ratio * 100.0:.1f}% "
            f"(required "
            f"{outlier_min_improvement_ratio * 100.0:.1f}%). "
            "Check marker placement, perspective, tape plane, "
            "or lens distortion."
        )

    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    H_normalized = current["H_normalized"]

    H_pixel_to_mm = current["H_pixel_to_mm"]

    magenta_distances = current["magenta_distances"]

    cyan_distances = current["cyan_distances"]

    valid_distances = np.concatenate(
        (
            magenta_distances[magenta_valid],
            cyan_distances[cyan_valid],
        )
    )

    if len(valid_distances) == 0:

        raise RuntimeError("No valid calibration intervals remain.")

    errors = np.abs(valid_distances - marker_spacing_mm)

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
        green_angle_deg=(green_angle_deg),
        excluded_segments=(excluded_segments),
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
            [
                width - 1.0,
                height - 1.0,
            ],
            [
                0.0,
                height - 1.0,
            ],
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

        label = f"M{i}-M{i + 1}"

        excluded = label in calibration.excluded_segments

        suffix = "  [EXCLUDED]" if excluded else ""

        print(f"M{i} -> M{i + 1}: " f"{distance:.3f} mm" f"{suffix}")

    print()
    print("CYAN:")

    for i, distance in enumerate(calibration.cyan_distances_mm):

        label = f"C{i}-C{i + 1}"

        excluded = label in calibration.excluded_segments

        suffix = "  [EXCLUDED]" if excluded else ""

        print(f"C{i} -> C{i + 1}: " f"{distance:.3f} mm" f"{suffix}")

    print()

    if calibration.excluded_segments:

        print("Excluded segments: " + ", ".join(calibration.excluded_segments))

    else:

        print("Excluded segments: none")

    print(f"Green angle: " f"{calibration.green_angle_deg:.4f}°")

    print(
        f"Mean spacing error "
        f"(valid segments): "
        f"{calibration.mean_spacing_error_mm:.4f} mm"
    )

    print(
        f"Maximum spacing error "
        f"(valid segments): "
        f"{calibration.max_spacing_error_mm:.4f} mm"
    )
