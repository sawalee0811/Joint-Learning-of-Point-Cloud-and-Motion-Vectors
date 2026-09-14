import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import imageio
import numpy as np
import open3d as o3d
import torch
from pytorch3d.renderer.points.pulsar import Renderer
from pytorch3d.transforms import matrix_to_axis_angle


# ===================== Config =====================

IMAGE_SIZE = 1024
WORLD_SCALE = 1.0 / 100

FAR_PLANE = 20.0
NEAR_PLANE = 0.0

VIEWS_PER_RING = 20

RUNS = [
    ("ims_1", 0.6, 0.5, "color"),
    ("ims_2", 0.6, 0.01, "color"),
]

FOCAL_SENSOR = (0.1, 0.2)
BG_COLOR = (0.0, 1.0, 0.0)

DIST_PLAN: Dict[float, List[str]] = {
    120.0: ["body"],
}

ELEV_PLAN: Dict[float, Dict[str, List[float]]] = {
    120.0: {
        "body": [60, 30, 0, -30, -60],
    },
}

K_DOWN = 0.05
K_UP = 0.05
MAX_ELEV_ANGLE = 30.0

DIST_SCALE_START = 35.0
DIST_SCALE = 0.05


# ===================== Utilities =====================

def _dist_tag(distance: float) -> str:
    return f"{int(round(distance)):04d}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_ply(
    path: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pcd = o3d.io.read_point_cloud(path)

    points = torch.from_numpy(
        np.asarray(pcd.points)
    ).float().to(device)

    colors = torch.from_numpy(
        np.asarray(pcd.colors)
    ).float().to(device)

    return points, colors


def compute_aabb_y(
    points: torch.Tensor,
) -> Tuple[float, float, float, float]:
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())

    height = y_max - y_min
    y_center = (y_min + y_max) * 0.5

    return y_min, y_max, y_center, height


def camera_on_sphere(
    x_center: float,
    y_center: float,
    z_center: float,
    distance: float,
    elevation_deg: float,
    azimuth_deg: float,
    device: torch.device,
) -> torch.Tensor:
    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)

    scaled_distance = distance * WORLD_SCALE
    radius_xy = scaled_distance * math.cos(elevation)

    x = x_center + radius_xy * math.sin(azimuth)
    y = y_center + scaled_distance * math.sin(elevation)
    z = z_center + radius_xy * math.cos(azimuth)

    return torch.tensor(
        [x, y, z],
        dtype=torch.float32,
        device=device,
    )


def rotation_look_at(
    camera_position: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    up = torch.tensor(
        [0.0, 1.0, 0.0],
        dtype=torch.float32,
        device=device,
    )

    z_direction = target - camera_position
    z_axis = z_direction / (
        torch.norm(z_direction) + 1e-8
    )

    forward = -z_axis

    right = torch.cross(
        up,
        forward,
        dim=0,
    )
    right = right / (
        torch.norm(right) + 1e-8
    )

    up = torch.cross(
        forward,
        right,
        dim=0,
    )
    up = up / (
        torch.norm(up) + 1e-8
    )

    rotation_matrix = torch.stack(
        [right, up, forward],
        dim=1,
    ).unsqueeze(0)

    return matrix_to_axis_angle(
        rotation_matrix
    )[0].to(device)


def pack_cam_params(
    position: torch.Tensor,
    rotation_axis_angle: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    focal_sensor = torch.tensor(
        FOCAL_SENSOR,
        dtype=torch.float32,
        device=device,
    )

    return torch.cat(
        [
            position,
            rotation_axis_angle,
            focal_sensor,
        ]
    )


def _base_ring_height(
    y_center: float,
    height: float,
    distance: float,
    ring: str,
) -> float:
    if ring == "body":
        return y_center

    raise ValueError(
        f"Unsupported combo: dist={distance}, ring={ring}"
    )


def _adaptive_target_y(
    y_center: float,
    height: float,
    distance: float,
    ring: str,
    elevation_deg: float,
) -> float:
    base = _base_ring_height(
        y_center,
        height,
        distance,
        ring,
    )

    if abs(elevation_deg) > 1e-6:
        amount = (
            min(
                abs(elevation_deg),
                MAX_ELEV_ANGLE,
            )
            / MAX_ELEV_ANGLE
        )

        if elevation_deg > 0:
            base += K_UP * height * amount
        else:
            base -= K_DOWN * height * amount

    return base


# ===================== Rendering =====================

def render_one_run(
    root: str,
    ply_id: str,
    points: torch.Tensor,
    colors: torch.Tensor,
    image_subdir: str,
    run_radius: float,
    gamma_value: float,
    color_mode: str,
    device: torch.device,
) -> List[Dict]:
    image_height = IMAGE_SIZE
    image_width = IMAGE_SIZE

    point_count = points.shape[0]

    renderer = Renderer(
        image_width,
        image_height,
        point_count,
        right_handed_system=True,
        n_track=100,
    ).to(device)

    vertex_radius = torch.full(
        (point_count,),
        float(run_radius) * WORLD_SCALE,
        dtype=torch.float32,
        device=device,
    )

    background_color = torch.tensor(
        BG_COLOR,
        dtype=torch.float32,
        device=device,
    )

    if color_mode == "black":
        render_colors = torch.zeros_like(
            colors
        )
    else:
        render_colors = colors

    x_center, y_center, z_center = [
        float(value.item())
        for value in points.mean(0)
    ]

    _, _, y_box_center, height = compute_aabb_y(
        points
    )

    all_runs = []

    for distance, rings in DIST_PLAN.items():
        for ring in rings:
            distance_tag = _dist_tag(
                distance
            )

            output_dir = os.path.join(
                root,
                image_subdir,
                f"dist_{distance_tag}_{ring}",
                ply_id,
            )

            ensure_dir(
                output_dir
            )

            camera_records = []

            image_id = 1
            camera_id = 1

            elevations = ELEV_PLAN.get(
                distance,
                {},
            ).get(
                ring,
                [0.0],
            )

            for elevation in elevations:
                target_y = _adaptive_target_y(
                    y_box_center,
                    height,
                    distance,
                    ring,
                    elevation,
                )

                target = torch.tensor(
                    [
                        x_center,
                        target_y,
                        z_center,
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                for view_index in range(
                    VIEWS_PER_RING
                ):
                    azimuth = (
                        360.0
                        * view_index
                        / VIEWS_PER_RING
                    )

                    camera_position = camera_on_sphere(
                        x_center,
                        target_y,
                        z_center,
                        distance,
                        elevation,
                        azimuth,
                        device,
                    )

                    camera_rotation = rotation_look_at(
                        camera_position,
                        target,
                        device,
                    )

                    camera_params = pack_cam_params(
                        camera_position,
                        camera_rotation,
                        device,
                    )

                    image = renderer(
                        points,
                        render_colors,
                        vertex_radius,
                        camera_params,
                        float(gamma_value),
                        float(FAR_PLANE),
                        float(NEAR_PLANE),
                        torch.clone(background_color),
                        None,
                        0.0,
                    )

                    output_png = os.path.join(
                        output_dir,
                        f"output_{image_id:03d}.png",
                    )

                    image_uint8 = (
                        image.detach()
                        .cpu()
                        .clamp(0, 1)
                        .mul(255)
                        .to(torch.uint8)
                        .numpy()
                    )

                    imageio.imsave(
                        output_png,
                        image_uint8,
                    )

                    camera_records.append(
                        {
                            "cam_id": f"{camera_id:04d}",
                            "distance": float(distance),
                            "ring": ring,
                            "azimuth_deg": float(azimuth),
                            "elevation_deg": float(elevation),
                            "target": [
                                x_center,
                                target_y,
                                z_center,
                            ],
                            "bg": "green",
                            "gamma": float(gamma_value),
                            "color_mode": color_mode,
                            "cam_params": (
                                camera_params
                                .detach()
                                .cpu()
                                .numpy()
                                .tolist()
                            ),
                        }
                    )

                    image_id += 1
                    camera_id += 1

            all_runs.append(
                {
                    "ims_dir": os.path.join(
                        image_subdir,
                        f"dist_{distance_tag}_{ring}",
                        ply_id,
                    ),
                    "radius": float(run_radius),
                    "gamma": float(gamma_value),
                    "distance": float(distance),
                    "ring": ring,
                    "cameras": camera_records,
                }
            )

    return all_runs


# ===================== Data Preparation =====================

def prepare_data(root: str) -> None:
    torch.manual_seed(1)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    ply_dir = os.path.join(
        root,
        "Ply",
    )

    cam_json_path = os.path.join(
        root,
        "cam.json",
    )

    ensure_dir(
        root
    )

    if not os.path.isdir(ply_dir):
        raise FileNotFoundError(
            f"PLY directory not found: {ply_dir}"
        )

    ply_files = [
        os.path.join(
            ply_dir,
            filename,
        )
        for filename in os.listdir(ply_dir)
        if filename.lower().endswith(".ply")
    ]

    ply_files.sort()

    print(
        f"Found {len(ply_files)} PLY files."
    )

    if len(ply_files) == 0:
        raise RuntimeError(
            f"No PLY files found in: {ply_dir}"
        )

    all_records = []

    for ply_path in ply_files:
        ply_id = os.path.splitext(
            os.path.basename(ply_path)
        )[0]

        print(
            f"[Processing] {ply_id}"
        )

        points, colors = load_ply(
            ply_path,
            device,
        )

        points = points * WORLD_SCALE

        print(
            f"Point count: {points.shape[0]}"
        )

        runs = []

        for (
            image_subdir,
            radius,
            gamma,
            color_mode,
        ) in RUNS:
            runs.extend(
                render_one_run(
                    root=root,
                    ply_id=ply_id,
                    points=points,
                    colors=colors,
                    image_subdir=image_subdir,
                    run_radius=radius,
                    gamma_value=gamma,
                    color_mode=color_mode,
                    device=device,
                )
            )

        all_records.append(
            {
                "ply_id": ply_id,
                "runs": runs,
            }
        )

    with open(
        cam_json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_records,
            file,
            indent=2,
        )

    print(
        f"Camera metadata saved to: {cam_json_path}"
    )


# ===================== Main =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Prepare rendered views and camera metadata "
            "for synthetic DPC training."
        )
    )

    parser.add_argument(
        "--data_base",
        type=str,
        default="path/to/your/data",
        help=(
            "Base directory containing the sequence folders."
        ),
    )

    args = parser.parse_args()

    DATA_BASE = args.data_base

    GOF = 3

    # Default example
    sequences = list("abcdef")
    categories = [
        "man",
        "woman",
    ]

    # Full experiment example
    # sequences = list("abcdefghijkl")
    # categories = [
    #     "man",
    #     "woman",
    # ]

    for category in categories:
        base_root = os.path.join(
            DATA_BASE,
            f"data_{category}{GOF}",
        )

        for sequence in sequences:
            run_name = (
                f"{sequence}_{category}{GOF}"
            )

            root_dir = os.path.join(
                base_root,
                run_name,
            )

            print(
                "======================================="
            )
            print(
                f"Starting data preparation: {root_dir}"
            )
            print(
                "======================================="
            )

            try:
                if not os.path.isdir(
                    root_dir
                ):
                    print(
                        f"Skipping {run_name}: "
                        f"directory not found: {root_dir}"
                    )
                else:
                    prepare_data(
                        root_dir
                    )

            except Exception as error:
                print(
                    f"Failed to prepare {run_name}: {error}"
                )

            print(
                f"Finished: {run_name}"
            )