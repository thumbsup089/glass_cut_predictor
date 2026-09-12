import os
import json
import sqlite3
import numpy as np

DB_PATH = "data/glass_data.db"
HEIGHTMAP_DIR = "data/heightmaps"


def init_database():
    """
    Erstellt Datenbank und Ordner, falls sie noch nicht existieren.
    """

    os.makedirs(HEIGHTMAP_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,
            stl_path TEXT,
            heightmap_path TEXT NOT NULL,

            width_mm REAL,
            height_mm REAL,
            max_depth_mm REAL,

            mean_slope REAL,
            max_slope REAL,

            mean_curvature REAL,
            max_curvature REAL,

            radial_vector TEXT,

            glass_thickness_mm REAL,
            burn_temperature_c REAL,
            hold_time_min REAL,

            coverage_percent REAL,

            notes TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_heightmap(heightmap, sample_name):
    """
    Speichert eine Heightmap als NumPy-Datei.

    NaN-Werte bleiben erhalten.
    """

    os.makedirs(HEIGHTMAP_DIR, exist_ok=True)

    filename = f"{sample_name}.npy"

    path = os.path.join(HEIGHTMAP_DIR, filename)

    np.save(path, heightmap)

    return path


def load_heightmap(heightmap_path):
    """
    Lädt eine gespeicherte Heightmap.
    """

    if not os.path.exists(heightmap_path):
        raise FileNotFoundError(f"Heightmap nicht gefunden: " f"{heightmap_path}")

    return np.load(heightmap_path)


def add_sample(
    name,
    stl_path,
    heightmap,
    feature_vector,
    radial_vector,
    glass_thickness_mm=None,
    burn_temperature_c=None,
    hold_time_min=None,
    coverage_percent=None,
    notes=None,
):
    """
    Fügt einen neuen Datensatz hinzu.

    feature_vector erwartet z.B.:

    {
        "width_mm": 220.0,
        "height_mm": 180.0,
        "max_depth_mm": 35.0,
        "mean_slope": 12.3,
        "max_slope": 41.2,
        "mean_curvature": 0.04,
        "max_curvature": 0.21
    }

    radial_vector:
        Liste mit 360 Radien in mm
    """

    init_database()

    heightmap_path = save_heightmap(heightmap, name)

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO samples (
            name,
            stl_path,
            heightmap_path,

            width_mm,
            height_mm,
            max_depth_mm,

            mean_slope,
            max_slope,

            mean_curvature,
            max_curvature,

            radial_vector,

            glass_thickness_mm,
            burn_temperature_c,
            hold_time_min,

            coverage_percent,

            notes
        )
        VALUES (
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?, ?,
            ?,
            ?
        )
    """,
        (
            name,
            stl_path,
            heightmap_path,
            feature_vector.get("width_mm"),
            feature_vector.get("height_mm"),
            feature_vector.get("max_depth_mm"),
            feature_vector.get("mean_slope"),
            feature_vector.get("max_slope"),
            feature_vector.get("mean_curvature"),
            feature_vector.get("max_curvature"),
            json.dumps(radial_vector),
            glass_thickness_mm,
            burn_temperature_c,
            hold_time_min,
            coverage_percent,
            notes,
        ),
    )

    sample_id = cursor.lastrowid

    conn.commit()
    conn.close()

    print(f"Sample gespeichert: " f"id={sample_id}, name={name}")

    return sample_id


def get_sample(sample_id):
    """
    Lädt einen Datensatz anhand seiner ID.
    """

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM samples
        WHERE id = ?
        """,
        (sample_id,),
    )

    row = cursor.fetchone()

    conn.close()

    if row is None:
        return None

    sample = dict(row)

    if sample["radial_vector"]:
        sample["radial_vector"] = json.loads(sample["radial_vector"])

    return sample


def get_all_samples():
    """
    Gibt alle Datensätze zurück.
    """

    init_database()

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM samples
        ORDER BY id ASC
    """)

    rows = cursor.fetchall()

    conn.close()

    samples = []

    for row in rows:

        sample = dict(row)

        if sample["radial_vector"]:
            sample["radial_vector"] = json.loads(sample["radial_vector"])

        samples.append(sample)

    return samples


def get_feature_vector(sample_id):
    """
    Gibt nur den Merkmalsvektor
    eines Samples zurück.
    """

    sample = get_sample(sample_id)

    if sample is None:
        return None

    return np.array(
        [
            sample["width_mm"],
            sample["height_mm"],
            sample["max_depth_mm"],
            sample["mean_slope"],
            sample["max_slope"],
            sample["mean_curvature"],
            sample["max_curvature"],
        ],
        dtype=float,
    )


def get_radial_vector(sample_id):
    """
    Gibt die gespeicherten Radien
    als NumPy-Array zurück.
    """

    sample = get_sample(sample_id)

    if sample is None:
        return None

    radial_vector = sample["radial_vector"]

    if radial_vector is None:
        return None

    return np.asarray(radial_vector, dtype=float)


def delete_sample(sample_id, delete_heightmap=True):
    """
    Löscht einen Datensatz.

    Optional wird auch die gespeicherte
    Heightmap-Datei entfernt.
    """

    sample = get_sample(sample_id)

    if sample is None:
        print(f"Sample {sample_id} " f"nicht gefunden.")
        return

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM samples
        WHERE id = ?
        """,
        (sample_id,),
    )

    conn.commit()
    conn.close()

    if delete_heightmap:

        path = sample["heightmap_path"]

        if path and os.path.exists(path):
            os.remove(path)

    print(f"Sample {sample_id} gelöscht.")


def print_samples():
    """
    Kleine Konsolenübersicht.
    """

    samples = get_all_samples()

    if not samples:
        print("Keine Samples " "in der Datenbank.")
        return

    for sample in samples:

        print(
            f"[{sample['id']}] "
            f"{sample['name']} | "
            f"depth="
            f"{sample['max_depth_mm']} mm | "
            f"coverage="
            f"{sample['coverage_percent']}"
        )


if __name__ == "__main__":

    init_database()

    print(f"Datenbank bereit: " f"{DB_PATH}")

    print_samples()
