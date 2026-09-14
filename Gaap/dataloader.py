import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

"""
Not work on no GUI's system! We need Gui windows to generate camera!
Input: initial point clouds amd number of frames you want to train
Output: json file named "all_meta.json" and "train.josn" record the camera parameters
"""

def load_ply_data(ply_file_path):
    # Load point cloud data from the PLY file
    pcd = o3d.io.read_point_cloud(str(ply_file_path))

    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty or cannot be loaded: {ply_file_path}")

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    return points, colors


def convert_matrices(extrinsic_1d, intrinsic_1d):
    # Convert intrinsic matrix from 1D to 3x3
    intrinsic_2d = np.array(
        [
            [intrinsic_1d[0], intrinsic_1d[1], intrinsic_1d[2]],
            [intrinsic_1d[3], intrinsic_1d[4], intrinsic_1d[5]],
            [intrinsic_1d[6], intrinsic_1d[7], intrinsic_1d[8]],
        ]
    )

    # Convert extrinsic matrix from 1D to 4x4
    extrinsic_2d = np.array(
        [
            [extrinsic_1d[0], extrinsic_1d[1], extrinsic_1d[2], extrinsic_1d[3]],
            [extrinsic_1d[4], extrinsic_1d[5], extrinsic_1d[6], extrinsic_1d[7]],
            [extrinsic_1d[8], extrinsic_1d[9], extrinsic_1d[10], extrinsic_1d[11]],
            [extrinsic_1d[12], extrinsic_1d[13], extrinsic_1d[14], extrinsic_1d[15]],
        ]
    )

    return intrinsic_2d, extrinsic_2d


def generate_camera_params(ply_file_path, x_values, y_values):
    # Load point cloud data
    points, colors = load_ply_data(ply_file_path)

    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    window_created = vis.create_window(
        width=600,
        height=600,
        visible=False,
    )

    if not window_created:
        raise RuntimeError(
            "Failed to create the Open3D visualization window."
        )

    k_list = []
    w2c_list = []
    cam_id_list = []
    fn_list = []

    image_count = 0

    # Generate camera parameters
    for y in y_values:
        for x in x_values:
            vis.clear_geometries()
            vis.add_geometry(pcd)

            render_option = vis.get_render_option()
            render_option.point_size = 1.0

            ctr = vis.get_view_control()
            ctr.rotate(x=x, y=y)

            vis.poll_events()
            vis.update_renderer()

            pinhole_params = ctr.convert_to_pinhole_camera_parameters()

            intrinsic_matrix = np.asarray(
                pinhole_params.intrinsic.intrinsic_matrix
            ).flatten()

            extrinsic_matrix = np.asarray(
                pinhole_params.extrinsic
            ).flatten()

            intrinsic_2d, extrinsic_2d = convert_matrices(
                extrinsic_matrix,
                intrinsic_matrix,
            )

            k_list.append(intrinsic_2d.tolist())
            w2c_list.append(extrinsic_2d.tolist())
            cam_id_list.append(image_count)
            fn_list.append(f"{image_count:03d}/frame1.png")

            image_count += 1

    vis.destroy_window()

    camera_params = {
        "w": 600,
        "h": 600,
        "k": k_list,
        "w2c": w2c_list,
        "cam_id": cam_id_list,
        "fn": fn_list,
    }

    return camera_params


def copy_parameters(camera_params, num_frames):
    # Copy camera parameters to all frames
    output_data = {
        "w": camera_params["w"],
        "h": camera_params["h"],
        "k": [],
        "w2c": [],
        "cam_id": [],
        "fn": [],
    }

    for frame_idx in range(1, num_frames + 1):
        k_new = camera_params["k"][:]
        w2c_new = camera_params["w2c"][:]
        cam_id_new = camera_params["cam_id"][:]

        fn_new = [
            fn.replace("frame1.png", f"frame{frame_idx}.png")
            for fn in camera_params["fn"]
        ]

        output_data["k"].append(k_new)
        output_data["w2c"].append(w2c_new)
        output_data["cam_id"].append(cam_id_new)
        output_data["fn"].append(fn_new)

    return output_data


def normalize_translation(data):
    # Normalize camera translation
    for i in range(len(data["w2c"])):
        for j in range(len(data["w2c"][i])):
            data["w2c"][i][j][0][3] /= 600
            data["w2c"][i][j][1][3] /= 600
            data["w2c"][i][j][2][3] /= 600

    return data


def save_metadata(data, output_dir):
    # Save training metadata
    output_path = output_dir / "train_meta.json"

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Training metadata saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training camera metadata."
    )

    parser.add_argument(
        "--ply_file",
        type=str,
        default="path/to/your/data/frame1.ply",
        help="Path to the initial PLY point cloud.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="path/to/your/output",
        help="Directory for the generated training metadata.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=30,
        help="Number of point cloud frames.",
    )

    args = parser.parse_args()

    ply_file_path = Path(args.ply_file)
    output_dir = Path(args.output_dir)

    if not ply_file_path.is_file():
        raise FileNotFoundError(f"PLY file not found: {ply_file_path}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_values = np.arange(0, 2200, 110)
    y_values = [0, 200, -200, 400, -400]

    camera_params = generate_camera_params(
        ply_file_path,
        x_values,
        y_values,
    )

    train_data = copy_parameters(
        camera_params,
        args.num_frames,
    )

    train_data = normalize_translation(
        train_data
    )

    save_metadata(
        train_data,
        output_dir,
    )