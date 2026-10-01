from pathlib import Path
import argparse
import json
import re
import shutil
import uuid


# =========================================================
# CONFIG
# =========================================================

BRAND_DIR = Path("data/Brandt001_14_09_2026")
METADATA_PATH = BRAND_DIR / "metadata.json"

FOLDERS_TO_FIX = [
    BRAND_DIR / "processed" / "before_calibrated_split",
    BRAND_DIR / "processed" / "after_calibrated_split",
    BRAND_DIR / "processed" / "after_calibrated_split_aligned",
]


# =========================================================
# FINAL MAPPING
#
# CURRENT wrong label -> FINAL correct label
#
# Based on the visual comparison of the six 2Eck/3Eck photos
# against the six STL silhouettes.
# =========================================================

FINAL_RENAME_MAP = {
    "VolumeToTop_glassUpper_UpperUpper_cut2Eck":
        "VolumeToTop_glassTop_cut2Eck",

    "VolumeToTop_glassTop_cut2Eck":
        "VolumeToTop_glassTop_cut3Eck",

    "VolumeToTop_glassTop_cut3Eck":
        "VolumeToTop_glassUpper_UpperUpper_cut3Eck",

    "VolumeToTop_glassUpper_UpperUpper_cut3Eck":
        "VolumeToTop_glassUpper_UpperUpper_cut2Eck",
}

# These two are already correct and are intentionally untouched:
UNCHANGED = {
    "VolumeToTop_glassLower_Upper_cut2Eck",
    "VolumeToTop_glassLower_Upper_cut3Eck",
}

ALL_FINAL_STEMS = set(FINAL_RENAME_MAP.values()) | UNCHANGED


# =========================================================
# EXPECTED BEFORE STATE
#
# Safety check using BEFORE image IDs.
# The script only applies if the folder matches the state
# shown in your current project.
# =========================================================

EXPECTED_CURRENT_BEFORE = {
    "VolumeToTop_glassLower_Upper_cut2Eck": "4",
    "VolumeToTop_glassUpper_UpperUpper_cut2Eck": "5",
    "VolumeToTop_glassTop_cut2Eck": "6",
    "VolumeToTop_glassUpper_UpperUpper_cut3Eck": "4",
    "VolumeToTop_glassLower_Upper_cut3Eck": "7",
    "VolumeToTop_glassTop_cut3Eck": "7",
}

EXPECTED_FINAL_BEFORE = {
    "VolumeToTop_glassLower_Upper_cut2Eck": "4",
    "VolumeToTop_glassTop_cut2Eck": "5",
    "VolumeToTop_glassTop_cut3Eck": "6",
    "VolumeToTop_glassUpper_UpperUpper_cut2Eck": "4",
    "VolumeToTop_glassLower_Upper_cut3Eck": "7",
    "VolumeToTop_glassUpper_UpperUpper_cut3Eck": "7",
}


# =========================================================
# HELPERS
# =========================================================

def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_metadata(metadata: dict):
    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")


def make_metadata_backup() -> Path:
    backup = METADATA_PATH.with_name(
        "metadata_before_final_label_fix.json"
    )

    index = 2

    while backup.exists():
        backup = METADATA_PATH.with_name(
            f"metadata_before_final_label_fix_{index}.json"
        )
        index += 1

    shutil.copy2(
        METADATA_PATH,
        backup,
    )

    return backup


def rel_to_brand(path: Path) -> str:
    return path.relative_to(
        BRAND_DIR
    ).as_posix()


def parse_image_id(filename: str) -> str | None:
    match = re.search(
        r"_img(\d+)_",
        filename,
    )

    return (
        match.group(1)
        if match
        else None
    )


def find_piece(
    metadata: dict,
    stem: str,
) -> dict:
    for piece in metadata.get(
        "fired_pieces",
        [],
    ):
        stl = piece.get("stl")

        if (
            stl
            and Path(stl).stem == stem
        ):
            return piece

    raise RuntimeError(
        f"No metadata entry found for: {stem}"
    )


def find_one_file(
    folder: Path,
    stem: str,
) -> Path:
    matches = [
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and stem in path.name
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one file for '{stem}' "
            f"in {folder}, found {len(matches)}."
        )

    return matches[0]


# =========================================================
# STATE DETECTION
# =========================================================

def get_before_state() -> dict[str, str | None]:
    folder = (
        BRAND_DIR
        / "processed"
        / "before_calibrated_split"
    )

    state = {}

    stems = (
        set(EXPECTED_CURRENT_BEFORE)
        | set(EXPECTED_FINAL_BEFORE)
    )

    for stem in stems:
        try:
            path = find_one_file(
                folder,
                stem,
            )

            state[stem] = parse_image_id(
                path.name
            )

        except RuntimeError:
            state[stem] = None

    return state


def state_matches(
    actual: dict,
    expected: dict,
) -> bool:
    return all(
        actual.get(stem)
        == image_id
        for stem, image_id
        in expected.items()
    )


# =========================================================
# RENAME
# =========================================================

def build_plan(
    folder: Path,
):
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    plan = []

    for path in sorted(
        folder.iterdir()
    ):
        if not path.is_file():
            continue

        matches = [
            old
            for old in FINAL_RENAME_MAP
            if old in path.name
        ]

        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous filename: {path.name}"
            )

        if not matches:
            continue

        old_label = matches[0]
        new_label = FINAL_RENAME_MAP[
            old_label
        ]

        destination = path.with_name(
            path.name.replace(
                old_label,
                new_label,
                1,
            )
        )

        plan.append(
            (
                path,
                destination,
            )
        )

    return plan


def validate_plan(plan):
    sources = {
        source.resolve()
        for source, _ in plan
    }

    destinations = [
        destination.resolve()
        for _, destination in plan
    ]

    if (
        len(destinations)
        != len(set(destinations))
    ):
        raise RuntimeError(
            "Rename plan creates duplicate filenames."
        )

    for _, destination in plan:
        if (
            destination.exists()
            and destination.resolve()
            not in sources
        ):
            raise FileExistsError(
                f"Destination already exists: {destination}"
            )


def apply_plan(plan):
    """
    Two-step rename through temporary filenames.
    This makes the 4-way cycle safe.
    """

    temporary_moves = []

    for source, destination in plan:
        temporary = source.with_name(
            f".__final_label_fix_{uuid.uuid4().hex}__"
            f"{source.name}"
        )

        source.rename(
            temporary
        )

        temporary_moves.append(
            (
                temporary,
                destination,
            )
        )

    for temporary, destination in temporary_moves:
        temporary.rename(
            destination
        )


# =========================================================
# METADATA
# =========================================================

def refresh_metadata(
    metadata: dict,
):
    before_dir = (
        BRAND_DIR
        / "processed"
        / "before_calibrated_split"
    )

    after_dir = (
        BRAND_DIR
        / "processed"
        / "after_calibrated_split"
    )

    for stem in sorted(
        ALL_FINAL_STEMS
    ):
        piece = find_piece(
            metadata,
            stem,
        )

        before_file = find_one_file(
            before_dir,
            stem,
        )

        after_file = find_one_file(
            after_dir,
            stem,
        )

        piece.setdefault(
            "before",
            {},
        )

        piece.setdefault(
            "after",
            {},
        )

        piece["before"]["crop"] = (
            rel_to_brand(
                before_file
            )
        )

        piece["after"]["crop"] = (
            rel_to_brand(
                after_file
            )
        )

        before_id = parse_image_id(
            before_file.name
        )

        after_id = parse_image_id(
            after_file.name
        )

        if before_id is not None:
            piece["before"][
                "image_id"
            ] = before_id

        if after_id is not None:
            piece["after"][
                "image_id"
            ] = after_id

        print(
            f"Metadata: {stem}"
        )
        print(
            f"  before -> "
            f"{piece['before']['crop']}"
        )
        print(
            f"  after  -> "
            f"{piece['after']['crop']}"
        )


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply the final verified 2Eck/3Eck photo-label mapping."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually perform changes. "
            "Without this flag: dry run only."
        ),
    )

    args = parser.parse_args()

    actual_state = get_before_state()

    if state_matches(
        actual_state,
        EXPECTED_FINAL_BEFORE,
    ):
        print(
            "Already in FINAL state. Nothing to do."
        )
        return

    if not state_matches(
        actual_state,
        EXPECTED_CURRENT_BEFORE,
    ):
        print(
            "STOP: project is not in the expected current state."
        )
        print()
        print(
            "Expected current BEFORE mapping:"
        )

        for stem, img in (
            EXPECTED_CURRENT_BEFORE.items()
        ):
            print(
                f"  {stem} -> img{img}"
            )

        print()
        print(
            "Actual detected BEFORE mapping:"
        )

        for stem in sorted(
            actual_state
        ):
            print(
                f"  {stem} -> "
                f"{'img' + actual_state[stem] if actual_state[stem] else 'MISSING'}"
            )

        print()
        print(
            "Nothing was changed."
        )
        return

    plans = {}
    all_moves = []

    for folder in FOLDERS_TO_FIX:
        plan = build_plan(
            folder
        )

        validate_plan(
            plan
        )

        plans[folder] = plan
        all_moves.extend(
            plan
        )

    print()
    print(
        "FINAL label-fix plan"
    )
    print(
        "===================="
    )

    for source, destination in all_moves:
        print(
            rel_to_brand(
                source
            )
        )
        print(
            "  -> "
            + rel_to_brand(
                destination
            )
        )

    print()
    print(
        "Unchanged:"
    )
    print(
        "  Lower_Upper_cut2Eck"
    )
    print(
        "  Lower_Upper_cut3Eck"
    )

    if not args.apply:
        print()
        print(
            "DRY RUN ONLY — nothing changed."
        )
        print()
        print(
            "If this looks correct, run:"
        )
        print(
            r"python .\fix_labels_final.py --apply"
        )
        return

    metadata = load_metadata()

    backup = make_metadata_backup()

    for folder in FOLDERS_TO_FIX:
        apply_plan(
            plans[
                folder
            ]
        )

    refresh_metadata(
        metadata
    )

    save_metadata(
        metadata
    )

    final_state = get_before_state()

    if not state_matches(
        final_state,
        EXPECTED_FINAL_BEFORE,
    ):
        raise RuntimeError(
            "Files were renamed, but final-state verification failed. "
            "Use the metadata backup and inspect the folders."
        )

    print()
    print(
        "DONE"
    )
    print(
        "===="
    )
    print(
        "Final mapping verified."
    )
    print(
        f"Metadata backup: {backup}"
    )
    print()
    print(
        "This script is idempotent: if you run it again, "
        "it will detect the FINAL state and do nothing."
    )


if __name__ == "__main__":
    main()
