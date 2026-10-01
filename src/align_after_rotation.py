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
OUTPUT_DIR = BRAND_DIR / "processed" / "after_calibrated_split_aligned"

ALPHA = 0.50

MOVE_STEP_PX = 1
MOVE_LARGE_STEP_PX = 10

ROTATE_FINE_DEG = 0.1
ROTATE_COARSE_DEG = 1.0


# =========================================================
# IO
# =========================================================


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return image


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
    for piece in metadata.get("fired_pieces", []):
        if piece.get("id") == piece_id:
            return piece

    raise KeyError(f"Piece not found in metadata: {piece_id}")


# =========================================================
# TRANSFORM
# =========================================================


def transform_after(
    after: np.ndarray,
    before_shape,
    angle_deg: float,
    dx_px: float,
    dy_px: float,
):
    """
    Rotate AFTER around its own center and translate it.

    The result is rendered into a canvas with exactly the
    same width and height as BEFORE.

    No scaling is applied.
    """

    before_h, before_w = before_shape[:2]
    after_h, after_w = after.shape[:2]

    center = (
        after_w / 2.0,
        after_h / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle_deg,
        1.0,
    )

    matrix[0, 2] += dx_px
    matrix[1, 2] += dy_px

    transformed = cv2.warpAffine(
        after,
        matrix,
        (before_w, before_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
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


# =========================================================
# PREVIEW
# =========================================================


def make_overlay(
    before: np.ndarray,
    after_transformed: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    BEFORE stays fixed.
    AFTER is shown at 50% opacity where AFTER actually exists.
    """

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


def add_hud(
    image: np.ndarray,
    angle_deg: float,
    dx_px: float,
    dy_px: float,
) -> np.ndarray:
    """
    Draw a compact, readable control panel with a dark background.
    """

    out = image.copy()

    lines = [
        f"Rotation: {angle_deg:.2f} deg",
        f"Move: dx={dx_px:.0f}px   dy={dy_px:.0f}px",
        "WASD = 1 px | Arrows = 10 px",
        "Q/E = 0.1 deg | Z/C = 1 deg",
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
        max_width = max(max_width, text_width)

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


def align_after_interactively(
    piece_id: str,
    before: np.ndarray,
    after: np.ndarray,
):
    dx_px = 0.0
    dy_px = 0.0
    angle_deg = 0.0

    window_name = f"Align AFTER to BEFORE - {piece_id}"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    # Open the window at exactly the BEFORE image dimensions.
    cv2.resizeWindow(
        window_name,
        before.shape[1],
        before.shape[0],
    )

    while True:
        after_transformed, valid_mask = transform_after(
            after=after,
            before_shape=before.shape,
            angle_deg=angle_deg,
            dx_px=dx_px,
            dy_px=dy_px,
        )

        preview = make_overlay(
            before=before,
            after_transformed=after_transformed,
            valid_mask=valid_mask,
        )

        preview = add_hud(
            image=preview,
            angle_deg=angle_deg,
            dx_px=dx_px,
            dy_px=dy_px,
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
                "after_aligned": after_transformed,
            }

        # 1 px movement
        if key in (ord("a"), ord("A")):
            dx_px -= MOVE_STEP_PX

        elif key in (ord("d"), ord("D")):
            dx_px += MOVE_STEP_PX

        elif key in (ord("w"), ord("W")):
            dy_px -= MOVE_STEP_PX

        elif key in (ord("s"), ord("S")):
            dy_px += MOVE_STEP_PX

        # Arrow keys: 10 px
        elif key in (2424832, 81):  # left
            dx_px -= MOVE_LARGE_STEP_PX

        elif key in (2555904, 83):  # right
            dx_px += MOVE_LARGE_STEP_PX

        elif key in (2490368, 82):  # up
            dy_px -= MOVE_LARGE_STEP_PX

        elif key in (2621440, 84):  # down
            dy_px += MOVE_LARGE_STEP_PX

        # Rotation
        elif key in (ord("q"), ord("Q")):
            angle_deg -= ROTATE_FINE_DEG

        elif key in (ord("e"), ord("E")):
            angle_deg += ROTATE_FINE_DEG

        elif key in (ord("z"), ord("Z")):
            angle_deg -= ROTATE_COARSE_DEG

        elif key in (ord("c"), ord("C")):
            angle_deg += ROTATE_COARSE_DEG

        elif key in (ord("r"), ord("R")):
            dx_px = 0.0
            dy_px = 0.0
            angle_deg = 0.0


# =========================================================
# PROCESS PIECE
# =========================================================


def process_piece(piece: dict):
    piece_id = piece["id"]

    before_path = BRAND_DIR / Path(piece["before"]["crop"])

    after_path = BRAND_DIR / Path(piece["after"]["crop"])

    before = load_image(before_path)
    after = load_image(after_path)

    print()
    print("========================================")
    print(piece_id)
    print("========================================")

    print(f"BEFORE: " f"{before.shape[1]} x {before.shape[0]} px")

    print(f"AFTER source: " f"{after.shape[1]} x {after.shape[0]} px")

    print(f"Output canvas: " f"{before.shape[1]} x {before.shape[0]} px")

    result = align_after_interactively(
        piece_id=piece_id,
        before=before,
        after=after,
    )

    if result is None:
        print("Cancelled.")
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / after_path.name

    success = cv2.imwrite(
        str(output_path),
        result["after_aligned"],
    )

    if not success:
        raise RuntimeError(f"Could not save: {output_path}")

    print()
    print(f"Saved: {output_path}")

    print(f"Rotation: " f"{result['angle_deg']:.2f} deg")

    print(
        f"Translation: " f"dx={result['dx_px']:.0f}px, " f"dy={result['dy_px']:.0f}px"
    )


# =========================================================
# CLI
# =========================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Manually align AFTER to BEFORE using " "translation and rotation only."
        )
    )

    parser.add_argument(
        "--piece",
        type=str,
        default=None,
        help=("Process one fired_piece ID. " "If omitted, process all fired pieces."),
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
        process_piece(piece)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
