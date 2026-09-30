import json
import re
from pathlib import Path

BRAND_DIR = Path("data/Brandt001_14_09_2026")
METADATA_PATH = BRAND_DIR / "metadata.json"
STL_DIR = BRAND_DIR / "stls"
BEFORE_DIR = BRAND_DIR / "processed" / "before_calibrated_split"
AFTER_DIR = BRAND_DIR / "processed" / "after_calibrated_split"
GLASS_THICKNESS_MM = 4.0


def rel(path: Path) -> str:
    return path.relative_to(BRAND_DIR).as_posix()


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"metadata.json not found: {METADATA_PATH}")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_stls() -> dict[str, Path]:
    if not STL_DIR.exists():
        raise FileNotFoundError(f"STL directory not found: {STL_DIR}")

    stls = {}

    for path in STL_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() != ".stl":
            continue

        key = path.stem.lower()

        if key in stls:
            raise RuntimeError(
                "Duplicate STL name ignoring upper/lower case:\n"
                f"  {stls[key].name}\n"
                f"  {path.name}"
            )

        stls[key] = path

    if not stls:
        raise RuntimeError(f"No STL files found in {STL_DIR}")

    return stls


def parse_crops(directory: Path, phase: str, brand_id: str) -> dict[str, dict]:
    if not directory.exists():
        raise FileNotFoundError(f"Crop directory not found: {directory}")

    pattern = re.compile(
        rf"^{re.escape(brand_id)}_{re.escape(phase)}_img(?P<image_id>\d+)_(?P<stl_name>.+)$",
        flags=re.IGNORECASE,
    )

    result = {}

    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue

        match = pattern.match(path.stem)

        if match is None:
            raise RuntimeError(
                f"Unexpected filename: {path.name}\n"
                f"Expected: {brand_id}_{phase}_img<number>_<STL-NAME>.png"
            )

        image_id = match.group("image_id")
        stl_name = match.group("stl_name")
        key = stl_name.lower()

        if key in result:
            raise RuntimeError(
                f"More than one {phase} crop for '{stl_name}':\n"
                f"  {result[key]['path'].name}\n"
                f"  {path.name}"
            )

        result[key] = {
            "path": path,
            "image_id": image_id,
            "stl_name": stl_name,
        }

    return result


def build_fired_pieces():
    metadata = load_metadata()
    brand_id = metadata.get("brand_id")

    if not brand_id:
        raise RuntimeError("metadata.json has no brand_id.")

    stls = find_stls()
    before_crops = parse_crops(BEFORE_DIR, "before", brand_id)
    after_crops = parse_crops(AFTER_DIR, "after", brand_id)

    errors = []

    for key, crop in before_crops.items():
        if key not in stls:
            errors.append(f"Before crop has no matching STL: {crop['path'].name}")

    for key, crop in after_crops.items():
        if key not in stls:
            errors.append(f"After crop has no matching STL: {crop['path'].name}")

    for key, stl_path in stls.items():
        if key not in before_crops:
            errors.append(f"Missing BEFORE crop for STL: {stl_path.name}")

        if key not in after_crops:
            errors.append(f"Missing AFTER crop for STL: {stl_path.name}")

    if errors:
        print("\n========================================")
        print("FIRED PIECES NOT WRITTEN")
        print("========================================\n")

        for error in errors:
            print(error)

        raise RuntimeError(
            f"{len(errors)} matching error(s) found. metadata.json was not modified."
        )

    fired_pieces = []

    for key, stl_path in sorted(stls.items(), key=lambda item: item[1].name.lower()):
        before = before_crops[key]
        after = after_crops[key]

        fired_pieces.append(
            {
                "id": f"{brand_id}_{stl_path.stem}",
                "stl": rel(stl_path),
                "glass_thickness_mm": GLASS_THICKNESS_MM,
                "before": {
                    "image_id": before["image_id"],
                    "crop": rel(before["path"]),
                },
                "after": {
                    "image_id": after["image_id"],
                    "crop": rel(after["path"]),
                },
            }
        )

    metadata["fired_pieces"] = fired_pieces

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\n========================================")
    print("FIRED PIECES COMPLETE")
    print("========================================")
    print(f"Brand: {brand_id}")
    print(f"STLs: {len(stls)}")
    print(f"Before crops: {len(before_crops)}")
    print(f"After crops: {len(after_crops)}")
    print(f"Fired pieces written: {len(fired_pieces)}\n")

    for piece in fired_pieces:
        print(
            f"OK  {piece['id']} "
            f"| before img{piece['before']['image_id']} "
            f"| after img{piece['after']['image_id']}"
        )

    print(f"\nUpdated: {METADATA_PATH}")


if __name__ == "__main__":
    build_fired_pieces()
