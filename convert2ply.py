import argparse
import os

import numpy as np
import torch


def write_ply_ascii_xyz_rgb(
    path,
    points_f32,
    colors_u8,
    extra_comments=None,
):
    """
    Write a point cloud to an ASCII PLY file.

    The output format contains:
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
    """
    num_points = points_f32.shape[0]

    assert points_f32.shape == (num_points, 3)
    assert colors_u8.shape == (num_points, 3)

    with open(path, "w") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")

        if extra_comments:
            for comment in extra_comments:
                file.write(f"comment {comment}\n")

        file.write(f"element vertex {num_points}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        for i in range(num_points):
            x, y, z = points_f32[i]
            r, g, b = colors_u8[i]

            file.write(
                f"{x:.6f} {y:.6f} {z:.6f} "
                f"{int(r)} {int(g)} {int(b)}\n"
            )


def convert_to_ply(
    params_path,
    output_folder,
    scale,
    start_idx=1,
):
    params_np = dict(
        np.load(params_path)
    )

    os.makedirs(
        output_folder,
        exist_ok=True,
    )

    params = {
        key: torch.tensor(value).float()
        for key, value in params_np.items()
    }

    if "means3D" not in params:
        raise KeyError(
            "'means3D' was not found in the parameter file."
        )

    if "rgb_colors" not in params:
        raise KeyError(
            "'rgb_colors' was not found in the parameter file."
        )

    num_frames = len(
        params["means3D"]
    )

    for frame_index in range(
        num_frames
    ):
        points = (
            params["means3D"][frame_index]
            * float(scale)
        ).numpy()

        colors = (
            params["rgb_colors"][frame_index]
        ).numpy()

        points = points.astype(
            np.float32
        )

        colors = np.clip(
            colors,
            0.0,
            1.0,
        )

        colors_u8 = np.rint(
            colors * 255.0
        ).astype(np.uint8)

        comments = [
            "Converted from learned point cloud parameters"
        ]

        output_path = os.path.join(
            output_folder,
            f"frame{start_idx + frame_index}.ply",
        )

        write_ply_ascii_xyz_rgb(
            output_path,
            points,
            colors_u8,
            extra_comments=comments,
        )

    print(
        f"Output PLY files saved to: {output_folder}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert learned point cloud parameters "
            "from NPZ format to ASCII PLY files."
        )
    )

    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to the params.npz file.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output PLY files.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=500.0,
        help=(
            "Scale factor used to recover the original "
            "point cloud coordinates."
        ),
    )

    parser.add_argument(
        "--start_idx",
        type=int,
        default=1,
        help="Starting frame index.",
    )

    args = parser.parse_args()

    convert_to_ply(
        params_path=args.params,
        output_folder=args.output_dir,
        scale=args.scale,
        start_idx=args.start_idx,
    )