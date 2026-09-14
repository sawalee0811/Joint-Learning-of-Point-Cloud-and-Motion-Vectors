import argparse
import os
import re

import numpy as np
from natsort import natsorted
from plyfile import PlyData


SCALE_SMALL_OBJECT = 100.0
SCALE_LARGE_OBJECT = 500.0


def load_first_ply(ply_dir: str) -> str:
    ply_files = [
        filename
        for filename in os.listdir(ply_dir)
        if filename.lower().endswith(".ply")
    ]

    if not ply_files:
        raise RuntimeError(
            f"No PLY files found in: {ply_dir}"
        )

    ply_files = natsorted(ply_files)

    return os.path.join(
        ply_dir,
        ply_files[0],
    )


def get_category_from_folder_name(
    folder_name: str,
) -> str:
    """
    Extract the sequence category from a case folder name.

    Examples:
        a_soldier3 -> soldier
        b_loot9 -> loot
        a_redandblack3 -> redandblack
        a_woman9 -> woman
        b_man3 -> man
    """
    parts = folder_name.split("_", 1)

    if len(parts) != 2:
        raise ValueError(
            f"Invalid folder name: {folder_name}. "
            f"Expected format such as a_soldier3."
        )

    sequence_name = parts[1]

    category = re.sub(
        r"\d+$",
        "",
        sequence_name,
    )

    if category == "":
        raise ValueError(
            f"Cannot parse category from folder name: "
            f"{folder_name}"
        )

    return category


def determine_scale(
    folder_name: str,
) -> float:
    """
    Use a scale of 100 for synthetic sequences and
    500 for real sequences.
    """
    category = get_category_from_folder_name(
        folder_name
    )

    if category in {
        "man",
        "woman",
    }:
        return SCALE_SMALL_OBJECT

    return SCALE_LARGE_OBJECT


def process_folder(
    folder_path: str,
) -> bool:
    ply_dir = os.path.join(
        folder_path,
        "Ply",
    )

    if not os.path.isdir(ply_dir):
        return False

    folder_name = os.path.basename(
        folder_path
    )

    ply_path = load_first_ply(
        ply_dir
    )

    scale = determine_scale(
        folder_name
    )

    print(
        f"[PROCESS] {ply_path} "
        f"(scale={scale})"
    )

    ply_data = PlyData.read(
        ply_path
    )

    vertex = ply_data["vertex"]

    xyz = np.vstack(
        [
            vertex["x"],
            vertex["y"],
            vertex["z"],
        ]
    ).T

    xyz = xyz / scale

    rgb = np.vstack(
        [
            vertex["red"],
            vertex["green"],
            vertex["blue"],
        ]
    ).T

    rgb = rgb / 255.0

    segmentation = np.ones(
        (xyz.shape[0], 1),
        dtype=np.float32,
    )

    final_data = np.hstack(
        [
            xyz,
            rgb,
            segmentation,
        ]
    )

    output_path = os.path.join(
        folder_path,
        "init_pt_cld.npz",
    )

    np.savez(
        output_path,
        data=final_data,
    )

    print(
        f"[SAVED] {output_path}"
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate init_pt_cld.npz files "
            "from the first PLY frame of each case."
        )
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=(
            "Root directory containing sequence case folders."
        ),
    )

    args = parser.parse_args()

    data_root = args.data_root

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Data root not found: {data_root}"
        )

    total_cases = 0
    success_cases = 0

    for name in sorted(
        os.listdir(data_root)
    ):
        folder_path = os.path.join(
            data_root,
            name,
        )

        if not os.path.isdir(folder_path):
            continue

        total_cases += 1

        try:
            processed = process_folder(
                folder_path
            )

            if processed:
                success_cases += 1

        except Exception as error:
            print(
                f"[ERROR] {folder_path}: {error}"
            )

    print("")
    print("=======================================")
    print(
        "[DONE] init_pt_cld.npz generation finished"
    )
    print(
        f"[DONE] total_cases = {total_cases}"
    )
    print(
        f"[DONE] success_cases = {success_cases}"
    )
    print("=======================================")


if __name__ == "__main__":
    main()