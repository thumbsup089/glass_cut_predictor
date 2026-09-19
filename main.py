import os
import cv2

from src.stl_to_heightmap import stl_to_heightmap
from src.radial_cut import calculate_radial_cut
from src.export_svg import export_radial_cut_svg

from src.marker_detection import (
    detect_markers,
    save_marker_debug_image,
)

from src.scale_image import (
    scale_image_to_pixels_per_mm,
)

from src.scale_image import (
    scale_image_to_pixels_per_mm,
    print_marker_spacing_report,
)

# ---------------------------------------------------------
# PHOTO CALIBRATION INPUT
# ---------------------------------------------------------

PHOTO_PATH = "data/Brandt1_14_09_2026/Davor/1.JPEG"

MARKER_SPACING_MM = 50.0

TARGET_PIXELS_PER_MM = 4.0


# ---------------------------------------------------------
# STL INPUT
# ---------------------------------------------------------

STL_PATH = "data/VolumeToTop_glassLower_Upper_cut1Eck.stl"

RESOLUTION_MM = 0.5

ANGLE_COUNT = 360


# ---------------------------------------------------------
# OUTPUT DIRECTORIES
# ---------------------------------------------------------

os.makedirs(
    "output/debug",
    exist_ok=True,
)

os.makedirs(
    "output/svg",
    exist_ok=True,
)


# =========================================================
# 1. DETECT PHOTO CALIBRATION MARKERS
# =========================================================

markers = detect_markers(PHOTO_PATH)

print()
print("Calibration markers detected:")

print(f"Magenta markers: " f"{len(markers.magenta_points)}")

print(f"Cyan markers:    " f"{len(markers.cyan_points)}")

print(f"Green pixels:    " f"{len(markers.green_pixels)}")


print()
print("Magenta points:")

for i, point in enumerate(markers.magenta_points):
    print(f"M{i}: {point}")


print()
print("Cyan points:")

for i, point in enumerate(markers.cyan_points):
    print(f"C{i}: {point}")


print_marker_spacing_report(
    "MAGENTA",
    markers.magenta_points,
    MARKER_SPACING_MM,
)

print_marker_spacing_report(
    "CYAN",
    markers.cyan_points,
    MARKER_SPACING_MM,
)


# =========================================================
# 2. SAVE MARKER DEBUG IMAGE
# =========================================================

save_marker_debug_image(
    PHOTO_PATH,
    markers,
    "output/debug/detected_markers.png",
)

print()
print("Marker debug image saved to " "output/debug/detected_markers.png")


# =========================================================
# 3. SCALE PHOTO
# =========================================================

scaled_image, scale_factor = scale_image_to_pixels_per_mm(
    PHOTO_PATH,
    markers.magenta_points,
    markers.cyan_points,
    marker_spacing_mm=MARKER_SPACING_MM,
    target_pixels_per_mm=TARGET_PIXELS_PER_MM,
)


SCALED_IMAGE_PATH = "output/debug/scaled_image.png"

cv2.imwrite(
    SCALED_IMAGE_PATH,
    scaled_image,
)

print()
print(f"Scaled image saved to " f"{SCALED_IMAGE_PATH}")

print(f"Scale factor: " f"{scale_factor:.6f}")


# =========================================================
# 4. EXISTING STL PIPELINE
# =========================================================

heightmap, xs, ys, origin_index = stl_to_heightmap(
    STL_PATH,
    resolution_mm=RESOLUTION_MM,
)


radial_cut = calculate_radial_cut(
    heightmap,
    pixel_size_mm=RESOLUTION_MM,
    center=origin_index,
    angle_count=ANGLE_COUNT,
)


# export_radial_cut_svg(
#     radial_cut,
#     "output/svg/radial_cut.svg",
# )


print()
print("Radial cut generated.")
