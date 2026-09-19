import os
import cv2

# =========================================================
# OPTIONAL STL PIPELINE - CURRENTLY DISABLED
# =========================================================

# from src.stl_to_heightmap import stl_to_heightmap
# from src.radial_cut import calculate_radial_cut
# from src.export_svg import export_radial_cut_svg


# =========================================================
# PHOTO CALIBRATION IMPORTS
# =========================================================

from src.marker_detection import (
    detect_markers,
    save_marker_debug_image,
)

from src.perspective_calibration import (
    calibrate_perspective,
    rectify_image,
    print_perspective_report,
)

from src.angle_detection import (
    detect_green_angle,
    save_angle_debug_image,
)

# =========================================================
# CONFIG
# =========================================================

# ---------------------------------------------------------
# PHOTO INPUT
# ---------------------------------------------------------

PHOTO_PATH = "data/Brandt1_14_09_2026/Davor/1.JPEG"

# Abstand zwischen zwei Magenta-/Cyan-Punkten
MARKER_SPACING_MM = 50.0

# Gewünschte spätere Bildauflösung
TARGET_PIXELS_PER_MM = 4.0


# ---------------------------------------------------------
# STL INPUT - CURRENTLY DISABLED
# ---------------------------------------------------------

# STL_PATH = "data/VolumeToTop_glassLower_Upper_cut1Eck.stl"

# RESOLUTION_MM = 0.5

# ANGLE_COUNT = 360


# =========================================================
# OUTPUT PATHS
# =========================================================

DEBUG_DIR = "output/debug"
SVG_DIR = "output/svg"


os.makedirs(
    DEBUG_DIR,
    exist_ok=True,
)

os.makedirs(
    SVG_DIR,
    exist_ok=True,
)


# =========================================================
# 1. DETECT COLOR MARKERS
# =========================================================

markers = detect_markers(PHOTO_PATH)


print()
print("========================================")
print("CALIBRATION MARKERS")
print("========================================")


print(f"Magenta markers: " f"{len(markers.magenta_points)}")

print(f"Cyan markers:    " f"{len(markers.cyan_points)}")

print(f"Green pixels:    " f"{len(markers.green_pixels)}")


# =========================================================
# PRINT MAGENTA POINTS
# =========================================================

print()
print("Magenta points:")


for i, point in enumerate(markers.magenta_points):
    print(f"M{i}: {point}")


# =========================================================
# PRINT CYAN POINTS
# =========================================================

print()
print("Cyan points:")


for i, point in enumerate(markers.cyan_points):
    print(f"C{i}: {point}")


# =========================================================
# 2. SAVE MARKER DEBUG IMAGE
# =========================================================

MARKER_DEBUG_PATH = f"{DEBUG_DIR}/detected_markers.png"


save_marker_debug_image(
    PHOTO_PATH,
    markers,
    MARKER_DEBUG_PATH,
)


print()
print(f"Marker debug image saved to: " f"{MARKER_DEBUG_PATH}")


# =========================================================
# 3. DETECT GREEN 90 DEGREE REFERENCE
# =========================================================

green_angle = detect_green_angle(markers.green_mask)


print()
print("========================================")
print("GREEN REFERENCE ANGLE")
print("========================================")


print(f"Corner: " f"{green_angle.corner}")


print(f"Angle visible in photo: " f"{green_angle.image_angle_deg:.3f}°")


print(f"Direction 1: " f"{green_angle.direction1}")


print(f"Direction 2: " f"{green_angle.direction2}")


# =========================================================
# 4. SAVE ANGLE DEBUG IMAGE
# =========================================================

ANGLE_DEBUG_PATH = f"{DEBUG_DIR}/detected_angle.png"


save_angle_debug_image(
    PHOTO_PATH,
    green_angle,
    ANGLE_DEBUG_PATH,
)


print()
print(f"Angle debug image saved to: " f"{ANGLE_DEBUG_PATH}")

# =========================================================
# 5. PERSPECTIVE CALIBRATION
# =========================================================

image = cv2.imread(PHOTO_PATH)

if image is None:
    raise FileNotFoundError(PHOTO_PATH)


perspective = calibrate_perspective(
    image_shape=image.shape,
    magenta_points=markers.magenta_points,
    cyan_points=markers.cyan_points,
    green_angle=green_angle,
    marker_spacing_mm=MARKER_SPACING_MM,
)


print_perspective_report(perspective)


# =========================================================
# 6. RECTIFY IMAGE
# =========================================================

rectified = rectify_image(
    image,
    perspective,
    pixels_per_mm=TARGET_PIXELS_PER_MM,
    padding_mm=10.0,
)


RECTIFIED_PATH = f"{DEBUG_DIR}/rectified.png"


cv2.imwrite(
    RECTIFIED_PATH,
    rectified,
)


print()
print(f"Rectified image saved to: " f"{RECTIFIED_PATH}")


# =========================================================
# 6. STL PIPELINE - CURRENTLY DISABLED
# =========================================================

# print()
# print("========================================")
# print("STL PIPELINE")
# print("========================================")
#
#
# heightmap, xs, ys, origin_index = (
#     stl_to_heightmap(
#         STL_PATH,
#         resolution_mm=RESOLUTION_MM,
#     )
# )
#
#
# radial_cut = calculate_radial_cut(
#     heightmap,
#     pixel_size_mm=RESOLUTION_MM,
#     center=origin_index,
#     angle_count=ANGLE_COUNT,
# )
#
#
# export_radial_cut_svg(
#     radial_cut,
#     f"{SVG_DIR}/radial_cut.svg",
# )
#
#
# print()
# print("Radial cut generated.")


# =========================================================
# DONE
# =========================================================

print()
print("========================================")
print("DONE")
print("========================================")
