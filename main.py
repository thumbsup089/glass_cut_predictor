from src.stl_to_heightmap import stl_to_heightmap
from src.radial_cut import calculate_radial_cut
from src.export_svg import export_radial_cut_svg

STL_PATH = "data/VolumeToTop_glassLower_Upper_cut1Eck.stl"

RESOLUTION_MM = 0.5
ANGLE_COUNT = 360


heightmap, xs, ys, origin_index = stl_to_heightmap(
    STL_PATH, resolution_mm=RESOLUTION_MM
)

radial_cut = calculate_radial_cut(
    heightmap, pixel_size_mm=RESOLUTION_MM, center=origin_index, angle_count=ANGLE_COUNT
)

export_radial_cut_svg(radial_cut, "output/svg/radial_cut.svg")

print("Radial cut generated.")
