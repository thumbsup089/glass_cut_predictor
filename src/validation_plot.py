from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_spacing_errors(
    magenta_distances_mm: list[float],
    cyan_distances_mm: list[float],
    target_mm: float = 50.0,
    output_path: str = "output/debug/spacing_errors.png",
):
    """
    Plot deviation of every measured segment
    from the expected marker spacing.
    """

    magenta = np.asarray(
        magenta_distances_mm,
        dtype=np.float64,
    )

    cyan = np.asarray(
        cyan_distances_mm,
        dtype=np.float64,
    )

    magenta_error = magenta - target_mm

    cyan_error = cyan - target_mm

    labels = [f"M{i}-M{i + 1}" for i in range(len(magenta))] + [
        f"C{i}-C{i + 1}" for i in range(len(cyan))
    ]

    errors = np.concatenate(
        (
            magenta_error,
            cyan_error,
        )
    )

    x = np.arange(len(errors))

    plt.figure(figsize=(11, 5))

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    plt.axhline(
        -0.5,
        linestyle="--",
        linewidth=1,
    )

    plt.plot(
        x,
        errors,
        marker="o",
    )

    plt.xticks(
        x,
        labels,
        rotation=45,
        ha="right",
    )

    plt.ylabel("Error from 50 mm [mm]")

    plt.xlabel("Marker segment")

    plt.title("Calibration spacing errors")

    plt.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def print_spacing_statistics(
    magenta_distances_mm: list[float],
    cyan_distances_mm: list[float],
    target_mm: float = 50.0,
):
    all_distances = np.asarray(
        magenta_distances_mm + cyan_distances_mm,
        dtype=np.float64,
    )

    errors = all_distances - target_mm

    print()
    print("========================================")
    print("SPACING ERROR STATISTICS")
    print("========================================")

    print(f"Mean measured spacing: " f"{np.mean(all_distances):.4f} mm")

    print(f"Mean signed error: " f"{np.mean(errors):+.4f} mm")

    print(f"Mean absolute error: " f"{np.mean(np.abs(errors)):.4f} mm")

    print(f"Standard deviation: " f"{np.std(errors, ddof=1):.4f} mm")

    print(f"Maximum absolute error: " f"{np.max(np.abs(errors)):.4f} mm")

    print()
    print("Individual errors:")

    index = 0

    for i, value in enumerate(magenta_distances_mm):
        error = value - target_mm

        print(f"M{i} -> M{i + 1}: " f"{value:.3f} mm " f"({error:+.3f} mm)")

        index += 1

    for i, value in enumerate(cyan_distances_mm):
        error = value - target_mm

        print(f"C{i} -> C{i + 1}: " f"{value:.3f} mm " f"({error:+.3f} mm)")


def plot_error_vs_radius(
    image_shape,
    magenta_points,
    cyan_points,
    magenta_distances_mm,
    cyan_distances_mm,
    target_mm: float = 50.0,
    output_path: str = "output/debug/error_vs_radius.png",
):
    """
    Plot calibration spacing error against distance
    from the original image center.

    A systematic relationship can indicate
    radial lens distortion.
    """

    height, width = image_shape[:2]

    image_center = np.array(
        [
            width / 2.0,
            height / 2.0,
        ],
        dtype=np.float64,
    )

    # Maximum useful radius:
    # image center -> corner
    max_radius = np.hypot(
        width / 2.0,
        height / 2.0,
    )

    all_radii = []
    all_errors = []
    all_labels = []
    all_groups = []

    def add_segments(
        name,
        points,
        distances_mm,
    ):
        pts = np.asarray(
            points,
            dtype=np.float64,
        )

        distances = np.asarray(
            distances_mm,
            dtype=np.float64,
        )

        for i in range(len(pts) - 1):
            midpoint = (pts[i] + pts[i + 1]) / 2.0

            radius_px = np.linalg.norm(midpoint - image_center)

            # Normalize:
            # 0 = image center
            # ~1 = image corner
            radius_normalized = radius_px / max_radius

            error_mm = distances[i] - target_mm

            all_radii.append(radius_normalized)

            all_errors.append(error_mm)

            all_labels.append(f"{name}{i}-{name}{i + 1}")

            all_groups.append(name)

    add_segments(
        "M",
        magenta_points,
        magenta_distances_mm,
    )

    add_segments(
        "C",
        cyan_points,
        cyan_distances_mm,
    )

    radii = np.asarray(
        all_radii,
        dtype=np.float64,
    )

    errors = np.asarray(
        all_errors,
        dtype=np.float64,
    )

    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(figsize=(9, 6))

    # Zero error
    plt.axhline(
        0.0,
        linewidth=1,
    )

    # ±0.5 mm reference
    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    plt.axhline(
        -0.5,
        linestyle="--",
        linewidth=1,
    )

    # Magenta tape
    magenta_indices = [i for i, group in enumerate(all_groups) if group == "M"]

    plt.scatter(
        radii[magenta_indices],
        errors[magenta_indices],
        label="Magenta tape",
    )

    # Cyan tape
    cyan_indices = [i for i, group in enumerate(all_groups) if group == "C"]

    plt.scatter(
        radii[cyan_indices],
        errors[cyan_indices],
        label="Cyan tape",
    )

    # Label every point
    for x, y, label in zip(
        radii,
        errors,
        all_labels,
    ):
        plt.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # =====================================================
    # SIMPLE TREND LINE
    # =====================================================

    if len(radii) >= 3:
        coefficients = np.polyfit(
            radii,
            errors,
            1,
        )

        x_fit = np.linspace(
            radii.min(),
            radii.max(),
            100,
        )

        y_fit = np.polyval(
            coefficients,
            x_fit,
        )

        plt.plot(
            x_fit,
            y_fit,
            linestyle=":",
            label="Linear trend",
        )

    plt.xlabel("Normalized distance from image center")

    plt.ylabel("Spacing error [mm]")

    plt.title("50 mm spacing error vs. image radius")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    # =====================================================
    # CONSOLE OUTPUT
    # =====================================================

    print()
    print("========================================")
    print("ERROR VS IMAGE RADIUS")
    print("========================================")

    for label, radius, error in zip(
        all_labels,
        radii,
        errors,
    ):
        print(f"{label}: " f"radius={radius:.3f}, " f"error={error:+.3f} mm")

    if len(radii) >= 2:
        correlation = np.corrcoef(
            radii,
            errors,
        )[0, 1]

        print()
        print(f"Radius/error correlation: " f"{correlation:+.3f}")
