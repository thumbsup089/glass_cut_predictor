from pathlib import Path
import argparse
import json

import cv2
import numpy as np

# =========================================================
# CONFIG
# =========================================================

BRAND_DIR = Path("data/Brandt001_14_09_2026")
METADATA_PATH = BRAND_DIR / "metadata.json"

AFTER_OUTPUT_DIR = BRAND_DIR / "processed" / "after_calibrated_split_aligned"

STL_MASK_OUTPUT_DIR = BRAND_DIR / "processed" / "stl_masks_aligned"

ALPHA = 0.50

# STL silhouettes must be mirrored left/right before alignment.
STL_MIRROR_HORIZONTAL = True

MOVE_STEP_PX = 1
MOVE_LARGE_STEP_PX = 10

ROTATE_FINE_DEG = 0.1
ROTATE_COARSE_DEG = 1.0


# =========================================================
# IO
# =========================================================


def load_color_image(path: Path) -> np.ndarray:
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return image


def load_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")

    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return mask


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def resolve_piece(
    metadata: dict,
    piece_id: str,
) -> dict:
    for piece in metadata.get(
        "fired_pieces",
        [],
    ):
        if piece.get("id") == piece_id:
            return piece

    raise KeyError(f"Piece not found in metadata: {piece_id}")


# =========================================================
# STL MASK PATH
# =========================================================


def find_stl_mask_path(piece: dict) -> Path:
    stl_rel = piece.get("stl")

    if not stl_rel:
        raise KeyError(f"{piece['id']}: no STL path in metadata.")

    stl_stem = Path(stl_rel).stem

    candidates = [
        (BRAND_DIR / "processed" / "stl_masks" / f"{stl_stem}_mask.png"),
        (
            BRAND_DIR
            / "processed"
            / "stl_silhouettes"
            / f"{stl_stem}_silhouette_mask.png"
        ),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)

    raise FileNotFoundError(
        f"{piece['id']}: STL mask not found.\n" f"Searched:\n{searched}"
    )


# =========================================================
# TRANSFORM
# =========================================================


def build_transform_matrix(
    source_shape,
    angle_deg: float,
    dx_px: float,
    dy_px: float,
) -> np.ndarray:
    source_h, source_w = source_shape[:2]

    center = (
        source_w / 2.0,
        source_h / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle_deg,
        1.0,
    )

    matrix[0, 2] += dx_px
    matrix[1, 2] += dy_px

    return matrix


def transform_after(
    after: np.ndarray,
    before_shape,
    angle_deg: float,
    dx_px: float,
    dy_px: float,
):
    before_h, before_w = before_shape[:2]
    after_h, after_w = after.shape[:2]

    matrix = build_transform_matrix(
        after.shape,
        angle_deg,
        dx_px,
        dy_px,
    )

    transformed = cv2.warpAffine(
        after,
        matrix,
        (before_w, before_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    valid_source = np.full(
        (after_h, after_w),
        255,
        dtype=np.uint8,
    )

    valid_mask = cv2.warpAffine(
        valid_source,
        matrix,
        (before_w, before_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return transformed, valid_mask


def transform_stl_mask(
    mask: np.ndarray,
    before_shape,
    angle_deg: float,
    dx_px: float,
    dy_px: float,
) -> np.ndarray:
    before_h, before_w = before_shape[:2]

    matrix = build_transform_matrix(
        mask.shape,
        angle_deg,
        dx_px,
        dy_px,
    )

    transformed = cv2.warpAffine(
        mask,
        matrix,
        (before_w, before_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    _, transformed = cv2.threshold(
        transformed,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return transformed


# =========================================================
# PREVIEW
# =========================================================


def make_after_overlay(
    before: np.ndarray,
    after_transformed: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    preview = before.copy()

    blended = cv2.addWeighted(
        before,
        1.0 - ALPHA,
        after_transformed,
        ALPHA,
        0.0,
    )

    valid = valid_mask > 0
    preview[valid] = blended[valid]

    return preview


def make_stl_mask_overlay(
    before: np.ndarray,
    mask_transformed: np.ndarray,
) -> np.ndarray:
    preview = before.copy()

    mask_pixels = mask_transformed > 0

    if np.any(mask_pixels):
        overlay_color = np.zeros_like(
            before,
            dtype=np.uint8,
        )

        # Cyan preview; saved mask stays pure black/white.
        overlay_color[:, :] = (
            255,
            255,
            0,
        )

        blended = cv2.addWeighted(
            before,
            1.0 - ALPHA,
            overlay_color,
            ALPHA,
            0.0,
        )

        preview[mask_pixels] = blended[mask_pixels]

        contours, _ = cv2.findContours(
            mask_transformed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        cv2.drawContours(
            preview,
            contours,
            -1,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return preview


def add_hud(
    image: np.ndarray,
    target: str,
    angle_deg: float,
    dx_px: float,
    dy_px: float,
    mirrored: bool,
) -> np.ndarray:
    out = image.copy()

    target_label = "AFTER" if target == "after" else "STL MASK"

    lines = [
        f"Target: {target_label}",
        f"Rotation: {angle_deg:.2f} deg",
        f"Move: dx={dx_px:.0f}px   dy={dy_px:.0f}px",
        (
            f"Mirror: {'ON' if mirrored else 'OFF'}"
            if target == "stl_mask"
            else "Mirror: n/a"
        ),
        "WASD = 1 px | Arrows = 10 px",
        "Q/E = 0.1 deg | Z/C = 1 deg",
        "M = mirror STL left/right",
        "R = reset | ENTER = save | ESC = cancel",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 30

    x = 15
    y_start = 32

    max_width = 0

    for line in lines:
        (text_width, _), _ = cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )

        max_width = max(
            max_width,
            text_width,
        )

    panel_x1 = 8
    panel_y1 = 8

    panel_x2 = min(
        image.shape[1] - 8,
        x + max_width + 18,
    )

    panel_y2 = min(
        image.shape[0] - 8,
        y_start + (len(lines) - 1) * line_height + 18,
    )

    panel = out.copy()

    cv2.rectangle(
        panel,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        panel,
        0.72,
        out,
        0.28,
        0,
        out,
    )

    for i, line in enumerate(lines):
        y = y_start + i * line_height

        cv2.putText(
            out,
            line,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return out


# =========================================================
# INTERACTIVE ALIGNMENT
# =========================================================


def align_interactively(
    piece_id: str,
    target: str,
    before: np.ndarray,
    source: np.ndarray,
):
    dx_px = 0.0
    dy_px = 0.0
    angle_deg = 0.0

    # STL masks start mirrored because your current silhouettes
    # need a left/right reflection to match the photographs.
    # Press M at any time to toggle this.
    mirrored = STL_MIRROR_HORIZONTAL if target == "stl_mask" else False

    target_label = "AFTER" if target == "after" else "STL MASK"

    window_name = f"Align {target_label} to BEFORE - {piece_id}"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        before.shape[1],
        before.shape[0],
    )

    while True:
        if target == "after":
            transformed, valid_mask = transform_after(
                after=source,
                before_shape=before.shape,
                angle_deg=angle_deg,
                dx_px=dx_px,
                dy_px=dy_px,
            )

            preview = make_after_overlay(
                before=before,
                after_transformed=transformed,
                valid_mask=valid_mask,
            )

        else:
            source_for_alignment = cv2.flip(source, 1) if mirrored else source

            transformed = transform_stl_mask(
                mask=source_for_alignment,
                before_shape=before.shape,
                angle_deg=angle_deg,
                dx_px=dx_px,
                dy_px=dy_px,
            )

            preview = make_stl_mask_overlay(
                before=before,
                mask_transformed=transformed,
            )

        preview = add_hud(
            image=preview,
            target=target,
            angle_deg=angle_deg,
            dx_px=dx_px,
            dy_px=dy_px,
            mirrored=mirrored,
        )

        cv2.imshow(
            window_name,
            preview,
        )

        key = cv2.waitKeyEx(0)

        if key == 27:
            cv2.destroyWindow(window_name)
            return None

        if key in (10, 13):
            cv2.destroyWindow(window_name)

            return {
                "angle_deg": angle_deg,
                "dx_px": dx_px,
                "dy_px": dy_px,
                "mirrored": mirrored,
                "aligned": transformed,
            }

        if key in (
            ord("a"),
            ord("A"),
        ):
            dx_px -= MOVE_STEP_PX

        elif key in (
            ord("d"),
            ord("D"),
        ):
            dx_px += MOVE_STEP_PX

        elif key in (
            ord("w"),
            ord("W"),
        ):
            dy_px -= MOVE_STEP_PX

        elif key in (
            ord("s"),
            ord("S"),
        ):
            dy_px += MOVE_STEP_PX

        elif key in (
            2424832,
            81,
        ):
            dx_px -= MOVE_LARGE_STEP_PX

        elif key in (
            2555904,
            83,
        ):
            dx_px += MOVE_LARGE_STEP_PX

        elif key in (
            2490368,
            82,
        ):
            dy_px -= MOVE_LARGE_STEP_PX

        elif key in (
            2621440,
            84,
        ):
            dy_px += MOVE_LARGE_STEP_PX

        elif key in (
            ord("q"),
            ord("Q"),
        ):
            angle_deg -= ROTATE_FINE_DEG

        elif key in (
            ord("e"),
            ord("E"),
        ):
            angle_deg += ROTATE_FINE_DEG

        elif key in (
            ord("z"),
            ord("Z"),
        ):
            angle_deg -= ROTATE_COARSE_DEG

        elif key in (
            ord("c"),
            ord("C"),
        ):
            angle_deg += ROTATE_COARSE_DEG

        elif target == "stl_mask" and key in (
            ord("m"),
            ord("M"),
        ):
            mirrored = not mirrored

        elif key in (
            ord("r"),
            ord("R"),
        ):
            dx_px = 0.0
            dy_px = 0.0
            angle_deg = 0.0

            if target == "stl_mask":
                mirrored = STL_MIRROR_HORIZONTAL


# =========================================================
# PROCESS PIECE
# =========================================================


def process_piece(
    piece: dict,
    target: str,
):
    piece_id = piece["id"]

    before_path = BRAND_DIR / Path(piece["before"]["crop"])

    before = load_color_image(before_path)

    if target == "after":
        source_path = BRAND_DIR / Path(piece["after"]["crop"])

        source = load_color_image(source_path)

        output_dir = AFTER_OUTPUT_DIR

        output_name = source_path.name

    else:
        source_path = find_stl_mask_path(piece)

        source = load_binary_mask(source_path)

        output_dir = STL_MASK_OUTPUT_DIR

        stl_stem = Path(piece["stl"]).stem

        output_name = f"{stl_stem}_mask.png"

    print()
    print("========================================")
    print(piece_id)
    print("========================================")

    print(f"Target: {target}")

    print(f"BEFORE: " f"{before.shape[1]} x " f"{before.shape[0]} px")

    print(f"Source: " f"{source.shape[1]} x " f"{source.shape[0]} px")

    print(f"Source path: " f"{source_path}")

    if target == "stl_mask":
        print(
            "STL mirror: "
            + ("horizontal (left/right)" if STL_MIRROR_HORIZONTAL else "off")
        )

    print(f"Output canvas: " f"{before.shape[1]} x " f"{before.shape[0]} px")

    result = align_interactively(
        piece_id=piece_id,
        target=target,
        before=before,
        source=source,
    )

    if result is None:
        print("Cancelled.")
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / output_name

    aligned = result["aligned"]

    if target == "stl_mask":
        _, aligned = cv2.threshold(
            aligned,
            127,
            255,
            cv2.THRESH_BINARY,
        )

    success = cv2.imwrite(
        str(output_path),
        aligned,
    )

    if not success:
        raise RuntimeError(f"Could not save: {output_path}")

    print()
    print(f"Saved: {output_path}")

    print(f"Rotation: " f"{result['angle_deg']:.2f} deg")

    print(
        f"Translation: " f"dx={result['dx_px']:.0f}px, " f"dy={result['dy_px']:.0f}px"
    )

    if target == "stl_mask":
        print("Mirrored left/right: " f"{'yes' if result['mirrored'] else 'no'}")


# =========================================================
# CLI
# =========================================================


def main():
    parser = argparse.ArgumentParser(
        description=("Align AFTER or STL MASK manually to BEFORE.")
    )

    parser.add_argument(
        "target",
        nargs="?",
        default="after",
        choices=[
            "after",
            "stl_mask",
        ],
        help=("What to align: after or stl_mask. " "Default: after."),
    )

    parser.add_argument(
        "--piece",
        type=str,
        default=None,
        help=("Process one fired_piece ID. " "If omitted, process all pieces."),
    )

    args = parser.parse_args()

    metadata = load_metadata()

    pieces = metadata.get(
        "fired_pieces",
        [],
    )

    if not pieces:
        raise RuntimeError("No fired_pieces found in metadata.json.")

    if args.piece:
        pieces = [
            resolve_piece(
                metadata,
                args.piece,
            )
        ]

    for piece in pieces:
        process_piece(
            piece=piece,
            target=args.target,
        )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
