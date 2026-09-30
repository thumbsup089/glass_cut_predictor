from pathlib import Path

import cv2

from src.metadata_manager import MetadataManager

from src.marker_detection import (
    detect_markers,
    save_marker_debug_image,
)

from src.angle_detection import (
    detect_green_angle,
    save_angle_debug_image,
)

from src.perspective_calibration import (
    calibrate_perspective,
    rectify_image,
)

from src.validation_plot import (
    plot_spacing_errors,
    print_spacing_statistics,
    plot_error_vs_radius,
)

# =========================================================
# BRAND CONFIG
# =========================================================

BRAND_DIR = Path("data/Brandt001_14_09_2026")

BRAND_ID = "B001"

BRAND_DATE = "2026-09-14"


# before oder after
PHASE = "before"


# =========================================================
# PATHS
# =========================================================

INPUT_DIR = BRAND_DIR / PHASE

OUTPUT_DIR = BRAND_DIR / "processed" / f"{PHASE}_calibrated"

DEBUG_DIR = BRAND_DIR / "processed" / "debug" / PHASE


# =========================================================
# CALIBRATION CONFIG
# =========================================================

MARKER_SPACING_MM = 50.0

TARGET_PIXELS_PER_MM = 4.0

PADDING_MM = 10.0

DEBUG = True


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


# =========================================================
# METADATA
# =========================================================

metadata = MetadataManager(
    brand_dir=BRAND_DIR,
    brand_id=BRAND_ID,
    date=BRAND_DATE,
)


# =========================================================
# FIND ORIGINAL + MARKED IMAGE PAIRS
# =========================================================


def get_image_pairs(
    directory: Path,
) -> list[tuple[Path, Path]]:

    if not directory.exists():

        raise FileNotFoundError(f"Input directory does not exist: {directory}")

    files = [
        file
        for file in directory.iterdir()
        if (file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)
    ]

    # Dateiname ohne Endung -> Pfad
    files_by_stem = {file.stem: file for file in files}

    pairs = []

    for marked_path in files:

        # Nur *_marked Dateien betrachten
        if not marked_path.stem.endswith("_marked"):
            continue

        # Beispiel:
        # 1_marked -> 1
        image_id = marked_path.stem.removesuffix("_marked")

        original_path = files_by_stem.get(image_id)

        if original_path is None:

            print(f"WARNING: No original image " f"found for {marked_path.name}")

            continue

        pairs.append(
            (
                original_path,
                marked_path,
            )
        )

    # Numerisch sortieren:
    # 1, 2, 3, 10 statt 1, 10, 2
    def sort_key(pair):

        stem = pair[0].stem

        try:
            return (0, int(stem))

        except ValueError:
            return (1, stem)

    return sorted(
        pairs,
        key=sort_key,
    )


# =========================================================
# PROCESS ONE IMAGE PAIR
# =========================================================


def process_image(
    original_path: Path,
    marked_path: Path,
):

    image_id = original_path.stem

    print()
    print("========================================")
    print(f"PROCESSING IMAGE {image_id}")
    print("========================================")

    print(f"Original: {original_path.name}")

    print(f"Marked:   {marked_path.name}")

    # -----------------------------------------------------
    # 1. LOAD ORIGINAL IMAGE
    # -----------------------------------------------------

    original_image = cv2.imread(str(original_path))

    if original_image is None:

        raise RuntimeError(f"Could not load original image: " f"{original_path}")

    # -----------------------------------------------------
    # 2. LOAD MARKED IMAGE
    # -----------------------------------------------------

    marked_image = cv2.imread(str(marked_path))

    if marked_image is None:

        raise RuntimeError(f"Could not load marked image: " f"{marked_path}")

    # -----------------------------------------------------
    # 3. DETECT MARKERS
    #
    # Marker werden NUR aus der marked-Datei erkannt.
    # -----------------------------------------------------

    markers = detect_markers(str(marked_path))

    print(f"Magenta markers: " f"{len(markers.magenta_points)}")

    print(f"Cyan markers:    " f"{len(markers.cyan_points)}")

    print(f"Green pixels:    " f"{len(markers.green_pixels)}")

    if len(markers.magenta_points) < 2:

        raise RuntimeError("Not enough magenta markers.")

    if len(markers.cyan_points) < 2:

        raise RuntimeError("Not enough cyan markers.")

    if len(markers.green_pixels) < 100:

        raise RuntimeError("Not enough green angle pixels.")

    # -----------------------------------------------------
    # 4. DETECT GREEN 90° REFERENCE
    # -----------------------------------------------------

    green_angle = detect_green_angle(markers.green_mask)

    print(f"Green angle in photo: " f"{green_angle.image_angle_deg:.3f}°")

    # -----------------------------------------------------
    # 5. CALCULATE PERSPECTIVE TRANSFORMATION
    #
    # Die Transformation wird anhand des
    # marked-Bildes berechnet.
    # -----------------------------------------------------

    calibration = calibrate_perspective(
        image_shape=marked_image.shape,
        magenta_points=markers.magenta_points,
        cyan_points=markers.cyan_points,
        green_angle=green_angle,
        marker_spacing_mm=MARKER_SPACING_MM,
    )

    print(f"Mean spacing error: " f"{calibration.mean_spacing_error_mm:.3f} mm")

    print(f"Maximum spacing error: " f"{calibration.max_spacing_error_mm:.3f} mm")

    print(f"Rectified green angle: " f"{calibration.green_angle_deg:.4f}°")

    # -----------------------------------------------------
    # 6. RECTIFY ORIGINAL IMAGE
    #
    # WICHTIG:
    # Nicht das marked-Bild wird entzerrt,
    # sondern das saubere Original.
    # -----------------------------------------------------

    rectified = rectify_image(
        marked_image,
        calibration,
        pixels_per_mm=TARGET_PIXELS_PER_MM,
        padding_mm=PADDING_MM,
    )

    # -----------------------------------------------------
    # 7. SAVE RECTIFIED IMAGE
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / f"{marked_path.stem}_rectified.png"

    success = cv2.imwrite(
        str(output_path),
        rectified,
    )

    if not success:

        raise RuntimeError(f"Could not save image: " f"{output_path}")

    print(f"Saved: {output_path}")

    # -----------------------------------------------------
    # 8. SAVE METADATA
    # -----------------------------------------------------

    original_relative = original_path.relative_to(BRAND_DIR).as_posix()

    marked_relative = marked_path.relative_to(BRAND_DIR).as_posix()

    rectified_relative = output_path.relative_to(BRAND_DIR).as_posix()

    metadata.add_or_update_source_image(
        image_id=image_id,
        phase=PHASE,
        original_path=original_relative,
        marked_path=marked_relative,
        rectified_path=rectified_relative,
        pixels_per_mm=TARGET_PIXELS_PER_MM,
        mean_error_mm=(calibration.mean_spacing_error_mm),
        max_error_mm=(calibration.max_spacing_error_mm),
        green_angle_deg=(calibration.green_angle_deg),
        excluded_segments=calibration.excluded_segments,
    )

    metadata.save()

    # =====================================================
    # OPTIONAL DEBUG
    # =====================================================

    if DEBUG:

        image_debug_dir = DEBUG_DIR / image_id

        image_debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------
        # Marker visualization
        # ---------------------------------------------

        save_marker_debug_image(
            str(marked_path),
            markers,
            str(image_debug_dir / "detected_markers.png"),
        )

        # ---------------------------------------------
        # Green angle visualization
        # ---------------------------------------------

        save_angle_debug_image(
            str(marked_path),
            green_angle,
            str(image_debug_dir / "detected_angle.png"),
        )

        # ---------------------------------------------
        # Numerical statistics
        # ---------------------------------------------

        print_spacing_statistics(
            calibration.magenta_distances_mm,
            calibration.cyan_distances_mm,
            target_mm=MARKER_SPACING_MM,
        )

        # ---------------------------------------------
        # Spacing error plot
        # ---------------------------------------------

        plot_spacing_errors(
            calibration.magenta_distances_mm,
            calibration.cyan_distances_mm,
            target_mm=MARKER_SPACING_MM,
            output_path=str(image_debug_dir / "spacing_errors.png"),
        )

        # ---------------------------------------------
        # Error vs image radius
        # ---------------------------------------------

        plot_error_vs_radius(
            image_shape=marked_image.shape,
            magenta_points=(markers.magenta_points),
            cyan_points=(markers.cyan_points),
            magenta_distances_mm=(calibration.magenta_distances_mm),
            cyan_distances_mm=(calibration.cyan_distances_mm),
            target_mm=MARKER_SPACING_MM,
            output_path=str(image_debug_dir / "error_vs_radius.png"),
        )

    return calibration


# =========================================================
# MAIN
# =========================================================


def main():

    image_pairs = get_image_pairs(INPUT_DIR)

    print()
    print("========================================")
    print("BATCH PHOTO RECTIFICATION")
    print("========================================")

    print(f"Brand: {BRAND_ID}")

    print(f"Phase: {PHASE}")

    print(f"Input directory: {INPUT_DIR}")

    print(f"Image pairs found: " f"{len(image_pairs)}")

    print(f"Target resolution: " f"{TARGET_PIXELS_PER_MM} px/mm")

    if not image_pairs:

        print("No image pairs found.")

        return

    successful = []

    failed = []

    # -----------------------------------------------------
    # PROCESS ALL IMAGE PAIRS
    # -----------------------------------------------------

    for (
        original_path,
        marked_path,
    ) in image_pairs:

        try:

            calibration = process_image(
                original_path,
                marked_path,
            )

            successful.append(
                (
                    original_path,
                    calibration,
                )
            )

        except Exception as error:

            failed.append(
                (
                    original_path,
                    str(error),
                )
            )

            print()

            print(f"FAILED: " f"{original_path.name}")

            print(f"Reason: {error}")

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("========================================")
    print("BATCH COMPLETE")
    print("========================================")

    print(f"Successful: " f"{len(successful)}")

    print(f"Failed:     " f"{len(failed)}")

    if successful:

        print()
        print("Successful images:")

        for (
            original_path,
            calibration,
        ) in successful:

            print(
                f"  {original_path.name}"
                f" | mean error "
                f"{calibration.mean_spacing_error_mm:.3f} mm"
                f" | max error "
                f"{calibration.max_spacing_error_mm:.3f} mm"
            )

    if failed:

        print()
        print("Failed images:")

        for (
            original_path,
            error,
        ) in failed:

            print(f"  {original_path.name}" f" | {error}")

    print()

    print(f"Output directory: " f"{OUTPUT_DIR}")

    print(f"Metadata: " f"{BRAND_DIR / 'metadata.json'}")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
