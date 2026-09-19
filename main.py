from src.stl_to_heightmap import stl_to_heightmap
from src.radial_cut import calculate_radial_cut
from src.export_svg import export_radial_cut_svg

from src.marker_detection import (
    detect_markers,
    save_marker_debug_image,
)

# ---------------------------------------------------------
# PHOTO CALIBRATION INPUT
# ---------------------------------------------------------

PHOTO_PATH = "data/Brandt1_14_09_2026/Davor/1.JPEG"

MARKER_SPACING_MM = 50.0


# ---------------------------------------------------------
# STL INPUT
# ---------------------------------------------------------

STL_PATH = "data/VolumeToTop_glassLower_Upper_cut1Eck.stl"

RESOLUTION_MM = 0.5

ANGLE_COUNT = 360


# =========================================================
# 1. DETECT PHOTO CALIBRATION MARKERS
# =========================================================

markers = detect_markers(PHOTO_PATH)

print()
print("Calibration markers detected:")
print(f"Magenta markers: {len(markers.magenta_points)}")
print(f"Cyan markers:    {len(markers.cyan_points)}")
print(f"Green pixels:    {len(markers.green_pixels)}")

print()
print("Magenta points:")

for i, point in enumerate(markers.magenta_points):
    print(f"M{i}: {point}")

print()
print("Cyan points:")

for i, point in enumerate(markers.cyan_points):
    print(f"C{i}: {point}")


save_marker_debug_image(
    PHOTO_PATH,
    markers,
    "output/debug/detected_markers.png",
)

print()
print("Marker debug image saved to " "output/debug/detected_markers.png")


# =========================================================
# 2. EXISTING STL PIPELINE
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
#    radial_cut,
#    "output/svg/radial_cut.svg",
# )

print()
print("Radial cut generated.")
