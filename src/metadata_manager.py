import json
from pathlib import Path


class MetadataManager:

    def __init__(
        self,
        brand_dir: Path,
        brand_id: str,
        date: str,
    ):
        self.brand_dir = Path(brand_dir)
        self.metadata_path = self.brand_dir / "metadata.json"

        self.brand_id = brand_id
        self.date = date

        self.data = self._load_or_create()

    # =====================================================
    # LOAD OR CREATE
    # =====================================================

    def _load_or_create(self):

        if self.metadata_path.exists():

            with open(
                self.metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)

        return {
            "brand_id": self.brand_id,
            "date": self.date,
            "notes": [],
            "images": [],
            "fired_pieces": [],
        }

    # =====================================================
    # ADD / UPDATE IMAGE
    # =====================================================

    def add_or_update_source_image(
        self,
        image_id: str,
        phase: str,
        original_path: str,
        marked_path: str,
        rectified_path: str,
        pixels_per_mm: float,
        mean_error_mm: float,
        max_error_mm: float,
        green_angle_deg: float,
        excluded_segments: list[str] | None = None,
    ):

        if excluded_segments is None:
            excluded_segments = []

        new_data = {
            "id": image_id,
            "phase": phase,
            "original": original_path,
            "marked": marked_path,
            "rectified": rectified_path,
            "calibration": {
                "pixels_per_mm": pixels_per_mm,
                "mean_error_mm": mean_error_mm,
                "max_error_mm": max_error_mm,
                "green_angle_deg": green_angle_deg,
                "excluded_segments": excluded_segments,
            },
        }

        for image in self.data["images"]:

            if image.get("id") == image_id and image.get("phase") == phase:
                image.update(new_data)
                return

        self.data["images"].append(new_data)

    # =====================================================
    # ADD / UPDATE FIRED PIECE
    # =====================================================

    def add_or_update_fired_piece(
        self,
        piece_id: str,
        before_path: str | None = None,
        after_path: str | None = None,
        negative_stl: str | None = None,
        glass_thickness_mm: float = 4.0,
    ):

        new_data = {
            "id": piece_id,
            "before": before_path,
            "after": after_path,
            "negative_stl": negative_stl,
            "glass_thickness_mm": glass_thickness_mm,
        }

        for piece in self.data["fired_pieces"]:

            if piece.get("id") == piece_id:
                piece.update(new_data)
                return

        self.data["fired_pieces"].append(new_data)

    # =====================================================
    # NOTES
    # =====================================================

    def add_note(
        self,
        note: str,
    ):

        if note not in self.data["notes"]:
            self.data["notes"].append(note)

    # =====================================================
    # SAVE
    # =====================================================

    def save(self):

        self.brand_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.data,
                file,
                indent=4,
                ensure_ascii=False,
            )
