#!/usr/bin/env python3

import argparse
import csv
import importlib
import json
import os
import random
import re
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from pytorch3d.renderer.points.pulsar.renderer import Renderer as PulsarRenderer
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from external import calc_ssim
from helpers import (
    l1_loss_v2,
    params2cpu,
    params2rendervar,
    save_params,
    weighted_l2_loss_v1,
    weighted_l2_loss_v2,
)


@dataclass(frozen=True)
class SequenceConfig:
    sequence: str
    setting_module: str
    camera_distance: float
    world_scale: float
    radius_phase1: float
    radius_phase2: float
    position_learning_rate: float
    phase1_gamma: float
    phase2_gamma: float
    amp_enabled: bool
    use_ssim_image_loss: bool
    detach_color_consistency: bool


def detect_sequence(data_path: str, exp_name: str) -> str:
    """
    Detect the sequence name from the data path or experiment name.
    """
    text = f"{data_path}/{exp_name}".lower().replace("\\", "/")

    sequence_names = [
        "redandblack",
        "longdress",
        "soldier",
        "woman",
        "loot",
        "man",
    ]

    matches = []

    for sequence in sequence_names:
        pattern = rf"(?<![a-z]){re.escape(sequence)}(?=\d|[^a-z]|$)"
        if re.search(pattern, text):
            matches.append(sequence)

    if len(matches) == 0:
        raise ValueError(
            "Cannot detect the sequence name from data_path or exp_name. "
            "Supported sequences are man, woman, redandblack, loot, "
            "longdress, and soldier."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple sequence names were detected: {matches}. "
            "Please check data_path and exp_name."
        )

    return matches[0]


def build_sequence_config(
    sequence: str,
    radius1: Optional[float],
    radius2: Optional[float],
) -> SequenceConfig:
    """
    Build all sequence-dependent settings.
    """
    synthetic_sequences = {"man", "woman"}
    is_synthetic = sequence in synthetic_sequences

    if radius1 is None:
        input_radius1 = 4.0 if is_synthetic else 2.0
    else:
        input_radius1 = radius1

    if radius2 is None:
        input_radius2 = 4.0 if is_synthetic else 2.0
    else:
        input_radius2 = radius2

    if input_radius1 <= 0.0 or input_radius2 <= 0.0:
        raise ValueError("radius1 and radius2 must be greater than zero.")

    if is_synthetic:
        radius_scale = 120.0 / 750.0
        setting_module = "pulsar_setting_synthetic"
        camera_distance = 120.0
        world_scale = 1.0 / 100.0
        position_learning_rate = 0.01
        phase1_gamma = 0.5
        amp_enabled = True
        use_ssim_image_loss = True
        detach_color_consistency = True

        if sequence == "man":
            phase2_gamma = 1e-2
        else:
            phase2_gamma = 1e-4
    else:
        radius_scale = 1.0
        setting_module = "pulsar_setting_real"
        camera_distance = 750.0
        world_scale = 1.0 / 500.0
        position_learning_rate = 0.0025
        phase2_gamma = 1e-4
        amp_enabled = False
        use_ssim_image_loss = False
        detach_color_consistency = False

        if sequence == "longdress":
            phase1_gamma = 1e-4
        else:
            phase1_gamma = 1.0

    return SequenceConfig(
        sequence=sequence,
        setting_module=setting_module,
        camera_distance=camera_distance,
        world_scale=world_scale,
        radius_phase1=input_radius1 * radius_scale,
        radius_phase2=input_radius2 * radius_scale,
        position_learning_rate=position_learning_rate,
        phase1_gamma=phase1_gamma,
        phase2_gamma=phase2_gamma,
        amp_enabled=amp_enabled,
        use_ssim_image_loss=use_ssim_image_loss,
        detach_color_consistency=detach_color_consistency,
    )


def load_setting_module(module_name: str) -> ModuleType:
    """
    Load the sequence-specific Pulsar setting module.
    """
    module = importlib.import_module(module_name)

    required_functions = [
        "initialize_params",
        "initialize_optimizer",
        "initialize_scheduler",
        "initialize_per_timestep",
        "initialize_post_first_timestep",
    ]

    missing_functions = [
        name for name in required_functions if not hasattr(module, name)
    ]

    if missing_functions:
        raise ImportError(
            f"{module_name} is missing required functions: "
            f"{', '.join(missing_functions)}"
        )

    return module


def _extract_frame_id_from_ims_dir(ims_dir: str) -> Optional[int]:
    """
    Extract the source frame ID from an image directory path.
    """
    match = re.search(r"frame(\d+)", str(ims_dir))

    if match is None:
        return None

    return int(match.group(1))


def _extract_frame_id_from_frame_record(
    frame_record: Dict[str, Any],
) -> Optional[int]:
    """
    Extract one source frame ID from one cam.json frame record.
    """
    frame_ids = []

    for run_record in frame_record.get("runs", []):
        frame_id = _extract_frame_id_from_ims_dir(
            run_record.get("ims_dir", "")
        )

        if frame_id is not None:
            frame_ids.append(frame_id)

    unique_ids = sorted(set(frame_ids))

    if len(unique_ids) == 0:
        return None

    if len(unique_ids) > 1:
        raise ValueError(
            "One cam.json frame record contains multiple source frame IDs: "
            f"{unique_ids}"
        )

    return unique_ids[0]


def load_cam_json(data_path: str) -> List[Dict[str, Any]]:
    """
    Load cam.json and sort records by their source frame IDs when available.
    """
    cam_json_path = os.path.join(data_path, "cam.json")

    if not os.path.isfile(cam_json_path):
        raise FileNotFoundError(f"cam.json not found: {cam_json_path}")

    with open(cam_json_path, "r", encoding="utf-8") as file:
        frames = json.load(file)

    if not isinstance(frames, list):
        raise ValueError("cam.json must contain a list of frame records.")

    if len(frames) == 0:
        return frames

    indexed_frames = []

    for original_index, frame_record in enumerate(frames):
        frame_id = _extract_frame_id_from_frame_record(frame_record)
        indexed_frames.append((frame_id, original_index, frame_record))

    available_ids = [
        frame_id
        for frame_id, _, _ in indexed_frames
        if frame_id is not None
    ]

    if len(available_ids) == 0:
        print(
            "Source frame IDs were not found in cam.json. "
            "The original cam.json order will be used."
        )
        return frames

    if len(available_ids) != len(frames):
        bad_indices = [
            original_index
            for frame_id, original_index, _ in indexed_frames
            if frame_id is None
        ]

        raise ValueError(
            "Source frame IDs are missing from some cam.json records: "
            f"{bad_indices[:10]}"
        )

    indexed_frames.sort(key=lambda item: (item[0], item[1]))

    sorted_frames = []

    for frame_id, original_index, frame_record in indexed_frames:
        frame_record["_real_frame_id"] = frame_id
        frame_record["_original_cam_json_index"] = original_index
        sorted_frames.append(frame_record)

    print("Loaded cam.json frames in numeric frame order.")

    return sorted_frames


def _image_index_from_path(path: str) -> int:
    """
    Extract the numeric index from an output image filename.
    """
    try:
        filename = os.path.splitext(os.path.basename(path))[0]
        number = filename.split("_")[-1]
        return int(number.lstrip("0") or "0")
    except (TypeError, ValueError):
        return -1


def _sorted_output_images(folder: str) -> List[str]:
    """
    Return output PNG files in numeric order.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Image folder not found: {folder}")

    files = [
        filename
        for filename in os.listdir(folder)
        if filename.lower().endswith(".png")
        and filename.startswith("output_")
    ]

    files.sort(key=_image_index_from_path)

    return [os.path.join(folder, filename) for filename in files]


def _read_image_as_float(path: str) -> torch.Tensor:
    """
    Read an image and return an RGB float tensor in the range [0, 1].
    """
    image = imageio.imread(path)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    if image.shape[2] == 4:
        image = image[..., :3]

    return torch.from_numpy(image.astype(np.float32) / 255.0)


def _compute_foreground_mask(image: torch.Tensor) -> torch.Tensor:
    """
    Create a foreground mask for images with a green background.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        green_mask = (
            (image[..., 0] == 0.0)
            & (image[..., 1] == 1.0)
            & (image[..., 2] == 0.0)
        )
        return (~green_mask).float()

    return torch.ones(image.shape[:2], dtype=torch.float32)


def _make_samples_for_run(
    data_path: str,
    run_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build image-camera samples for one run record.
    """
    image_directory = run_record["ims_dir"]
    folder = os.path.join(data_path, image_directory)

    image_paths = _sorted_output_images(folder)
    cameras = run_record.get("cameras", [])
    sample_count = min(len(image_paths), len(cameras))

    if sample_count == 0:
        raise RuntimeError(
            f"No valid image-camera pairs were found in: {folder}"
        )

    samples = []

    for index in range(sample_count):
        camera = cameras[index]
        image_path = image_paths[index]
        ground_truth = _read_image_as_float(image_path)

        samples.append(
            {
                "img_path": image_path,
                "gt_cpu": ground_truth,
                "fg_mask_cpu": _compute_foreground_mask(ground_truth),
                "cam_params": torch.tensor(
                    camera["cam_params"],
                    dtype=torch.float32,
                ),
                "distance": float(
                    run_record.get(
                        "distance",
                        camera.get("distance", -1.0),
                    )
                ),
                "ring": str(
                    run_record.get(
                        "ring",
                        camera.get("ring", ""),
                    )
                ),
                "elevation_deg": float(
                    camera.get("elevation_deg", 0.0)
                ),
                "azimuth_deg": float(
                    camera.get("azimuth_deg", 0.0)
                ),
                "cam_id": str(
                    camera.get("cam_id", f"{index + 1:04d}")
                ),
                "real_frame_id": _extract_frame_id_from_ims_dir(
                    image_directory
                ),
            }
        )

    return samples


def _float_equal(
    value1: float,
    value2: float,
    epsilon: float = 1e-3,
) -> bool:
    """
    Compare two floating-point values.
    """
    try:
        return abs(float(value1) - float(value2)) < epsilon
    except (TypeError, ValueError):
        return False


def get_phase_dataset(
    data_path: str,
    frame_record: Dict[str, Any],
    phase: str,
    camera_distance: float,
) -> List[Dict[str, Any]]:
    """
    Build the dataset for the movement or refinement phase.
    """
    if phase not in {"move", "refine"}:
        raise ValueError(f"Unsupported phase: {phase}")

    target_directory = "ims_1" if phase == "move" else "ims_2"
    samples = []

    for run_record in frame_record.get("runs", []):
        image_directory = run_record.get("ims_dir", "")

        is_target_directory = (
            image_directory.startswith(target_directory + os.sep)
            or image_directory.startswith(target_directory + "/")
        )

        if not is_target_directory:
            continue

        samples.extend(_make_samples_for_run(data_path, run_record))

    samples = [
        sample
        for sample in samples
        if _float_equal(
            sample.get("distance", -1.0),
            camera_distance,
        )
    ]

    def camera_id_to_int(camera_id: str) -> int:
        try:
            return int(str(camera_id).lstrip("0") or "0")
        except (TypeError, ValueError):
            return 10**9

    samples.sort(
        key=lambda sample: (
            sample["distance"],
            float(sample.get("elevation_deg", 0.0)),
            camera_id_to_int(sample["cam_id"]),
        )
    )

    return samples


def _amp_cast(
    tensor: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """
    Cast a tensor to float16 when AMP is enabled.
    """
    if enabled:
        return tensor.to(torch.float16)

    return tensor


def _get_radius_tensor(
    variables: Dict[str, Any],
    points: torch.Tensor,
    point_radius: float,
    world_scale: float,
) -> torch.Tensor:
    """
    Cache the radius tensor to reduce repeated allocations.
    """
    cache_key = (
        f"radius_tensor_"
        f"{float(point_radius):.10g}_"
        f"{float(world_scale):.10g}"
    )

    radius_tensor = variables.get(cache_key)

    if (
        radius_tensor is None
        or not isinstance(radius_tensor, torch.Tensor)
        or radius_tensor.shape[0] != points.shape[0]
        or radius_tensor.device != points.device
        or radius_tensor.dtype != points.dtype
    ):
        radius_tensor = torch.full(
            (points.shape[0],),
            float(point_radius) * float(world_scale),
            device=points.device,
            dtype=points.dtype,
        )
        variables[cache_key] = radius_tensor

    return radius_tensor


def _masked_l1(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate foreground L1 loss.
    """
    if foreground_mask.ndim == 3:
        foreground_mask = foreground_mask.mean(dim=-1)

    mask = (foreground_mask > 0.5).to(prediction.dtype)
    denominator = (mask.sum() * 3.0).clamp_min(1e-6)

    return (
        (prediction - ground_truth).abs()
        * mask.unsqueeze(-1)
    ).sum() / denominator


def render_prediction(
    params: Dict[str, torch.Tensor],
    variables: Dict[str, Any],
    camera_params: torch.Tensor,
    renderer: PulsarRenderer,
    gamma: float,
    point_radius: float,
    config: SequenceConfig,
) -> torch.Tensor:
    """
    Render one predicted image.
    """
    render_variables = params2rendervar(params)
    points = render_variables["means3D"]
    colors = render_variables["colors_precomp"]

    radius_tensor = _get_radius_tensor(
        variables=variables,
        points=points,
        point_radius=point_radius,
        world_scale=config.world_scale,
    )

    background_color = torch.tensor(
        [0.0, 1.0, 0.0],
        dtype=points.dtype,
        device=points.device,
    )

    camera_params = camera_params.to(
        device=points.device,
        dtype=torch.float32,
    )

    with autocast(enabled=False):
        prediction = renderer(
            points,
            colors,
            radius_tensor,
            camera_params,
            float(gamma),
            20.0,
            0.0,
            background_color,
            None,
            0.0,
        )

    return _amp_cast(prediction, config.amp_enabled)


def calculate_loss(
    params: Dict[str, torch.Tensor],
    sample: Dict[str, Any],
    variables: Dict[str, Any],
    renderer: PulsarRenderer,
    gamma: float,
    point_radius: float,
    phase: str,
    enable_geometry_losses: bool,
    config: SequenceConfig,
) -> Tuple[
    torch.Tensor,
    Dict[str, torch.Tensor],
    torch.Tensor,
]:
    """
    Calculate the training loss for one sample.
    """
    prediction = render_prediction(
        params=params,
        variables=variables,
        camera_params=sample["cam_params"],
        renderer=renderer,
        gamma=gamma,
        point_radius=point_radius,
        config=config,
    )

    ground_truth = sample["gt_cpu"].to(
        prediction.device,
        non_blocking=True,
    )

    if ground_truth.ndim == 2:
        ground_truth = ground_truth.unsqueeze(-1).repeat(1, 1, 3)

    losses: Dict[str, torch.Tensor] = {}

    if config.use_ssim_image_loss:
        full_l1 = (prediction - ground_truth).abs().mean()
        ssim_loss = 1.0 - calc_ssim(
            prediction.float(),
            ground_truth.float(),
        )
        losses["im"] = 0.8 * full_l1 + 0.2 * ssim_loss
    else:
        ground_truth = _amp_cast(
            ground_truth,
            config.amp_enabled,
        )

        full_l1 = (prediction - ground_truth).abs().mean()

        if phase == "refine":
            foreground_l1 = _masked_l1(
                prediction=prediction,
                ground_truth=ground_truth,
                foreground_mask=sample["fg_mask_cpu"].to(
                    prediction.device
                ),
            )
            losses["im"] = (
                0.8 * foreground_l1
                + 0.2 * full_l1
            )
        else:
            losses["im"] = full_l1

    render_variables = params2rendervar(params)
    points = render_variables["means3D"]

    if enable_geometry_losses:
        neighbor_points = points[variables["neighbor_indices"]]
        current_offset = neighbor_points - points[:, None]

        losses["rigid"] = weighted_l2_loss_v2(
            current_offset,
            variables["prev_offset"],
            variables["neighbor_weight"],
        )

        current_offset_magnitude = torch.sqrt(
            (current_offset**2).sum(dim=-1) + 1e-20
        )

        losses["iso"] = weighted_l2_loss_v1(
            current_offset_magnitude,
            variables["neighbor_dist"],
            variables["neighbor_weight"],
        )

        if config.detach_color_consistency:
            with torch.no_grad():
                losses["soft_col_cons"] = l1_loss_v2(
                    params["rgb_colors"],
                    variables["prev_col"],
                )
        else:
            losses["soft_col_cons"] = l1_loss_v2(
                params["rgb_colors"],
                variables["prev_col"],
            )
    else:
        zero = torch.tensor(
            0.0,
            device=points.device,
            dtype=points.dtype,
        )
        losses["rigid"] = zero
        losses["iso"] = zero
        losses["soft_col_cons"] = zero

    total_loss = (
        4.0 * losses["im"]
        + 1.0 * losses["rigid"]
        + 1.0 * losses["iso"]
        + 0.01 * losses["soft_col_cons"]
    )

    return total_loss, losses, prediction


def save_comparison_image(
    params: Dict[str, torch.Tensor],
    variables: Dict[str, Any],
    sample: Dict[str, Any],
    renderer: PulsarRenderer,
    output_path: str,
    gamma: float,
    point_radius: float,
    config: SequenceConfig,
) -> None:
    """
    Save a side-by-side predicted and ground-truth image.
    """
    with torch.no_grad():
        prediction = render_prediction(
            params=params,
            variables=variables,
            camera_params=sample["cam_params"],
            renderer=renderer,
            gamma=gamma,
            point_radius=point_radius,
            config=config,
        )

    prediction_image = (
        prediction.detach()
        .cpu()
        .clamp(0, 1)
        .float()
        .mul(255.0)
        .to(torch.uint8)
        .numpy()
    )

    ground_truth_image = (
        sample["gt_cpu"].cpu().numpy() * 255.0
    ).astype(np.uint8)

    if prediction_image.ndim == 2:
        prediction_image = np.repeat(
            prediction_image[..., None],
            3,
            axis=2,
        )

    if ground_truth_image.ndim == 2:
        ground_truth_image = np.repeat(
            ground_truth_image[..., None],
            3,
            axis=2,
        )

    comparison = np.concatenate(
        [prediction_image, ground_truth_image],
        axis=1,
    )

    imageio.imwrite(output_path, comparison)


def build_training_batch(
    dataset: Sequence[Dict[str, Any]],
    batch_size: int = 5,
) -> List[Dict[str, Any]]:
    """
    Build a batch with samples from different elevations.
    """
    if len(dataset) == 0:
        return []

    elevation_groups: Dict[int, List[Dict[str, Any]]] = {}

    for sample in dataset:
        elevation = int(
            round(float(sample.get("elevation_deg", 0.0)))
        )
        elevation_groups.setdefault(elevation, []).append(sample)

    elevations = sorted(
        elevation_groups.keys(),
        key=lambda value: (-abs(value), -value),
    )

    batch = []

    for elevation in elevations:
        if len(batch) >= batch_size:
            break

        batch.append(
            random.choice(elevation_groups[elevation])
        )

    index = 0

    while len(batch) < batch_size:
        elevation = elevations[index % len(elevations)]
        batch.append(
            random.choice(elevation_groups[elevation])
        )
        index += 1

    return batch[:batch_size]


def build_refinement_batch(
    dataset: Sequence[Dict[str, Any]],
    batch_size: int = 5,
) -> List[Dict[str, Any]]:
    """
    Build a refinement batch without replacement when possible.
    """
    if len(dataset) == 0:
        return []

    if len(dataset) >= batch_size:
        return random.sample(list(dataset), batch_size)

    batch = list(dataset)

    while len(batch) < batch_size:
        batch.append(random.choice(dataset))

    return batch


def pick_fixed_debug_sample(
    dataset: Sequence[Dict[str, Any]],
    camera_distance: float,
    view_index: int = 41,
    view_ring: str = "body",
) -> Optional[Dict[str, Any]]:
    """
    Select a stable camera sample for debug images.
    """
    if len(dataset) == 0:
        return None

    candidates = [
        sample
        for sample in dataset
        if _float_equal(
            sample.get("distance", -1.0),
            camera_distance,
        )
        and str(sample.get("ring", "")) == view_ring
    ]

    if len(candidates) == 0:
        fallback = random.choice(dataset)
        tqdm.write(
            f"[Debug] Fixed view fallback: "
            f"{fallback.get('img_path', '(no path)')}"
        )
        return fallback

    exact_image_matches = [
        sample
        for sample in candidates
        if _image_index_from_path(
            sample.get("img_path", "")
        )
        == view_index
    ]

    if exact_image_matches:
        return exact_image_matches[0]

    exact_camera_matches = []

    for sample in candidates:
        try:
            camera_id = int(
                str(sample.get("cam_id", "")).lstrip("0") or "0"
            )
        except (TypeError, ValueError):
            camera_id = -1

        if camera_id == view_index:
            exact_camera_matches.append(sample)

    if exact_camera_matches:
        return exact_camera_matches[0]

    fallback_index = min(
        view_index - 1,
        len(candidates) - 1,
    )

    return candidates[fallback_index]


def run_parallel_wave(
    params: Dict[str, torch.Tensor],
    samples: Sequence[Dict[str, Any]],
    variables: Dict[str, Any],
    renderers: Sequence[PulsarRenderer],
    streams: Sequence[torch.cuda.Stream],
    gamma: float,
    point_radius: float,
    phase: str,
    enable_geometry_losses: bool,
    config: SequenceConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Render and evaluate one parallel wave.
    """
    if len(renderers) < len(samples) or len(streams) < len(samples):
        raise ValueError(
            "The renderer and stream pools are smaller than the wave."
        )

    device = params["means3D"].device

    component_sum = {
        "im": torch.tensor(0.0, device=device),
        "rigid": torch.tensor(0.0, device=device),
        "iso": torch.tensor(0.0, device=device),
        "soft_col_cons": torch.tensor(0.0, device=device),
    }

    results = [None] * len(samples)

    with autocast(
        enabled=config.amp_enabled,
        dtype=torch.float16,
    ):
        for index, sample in enumerate(samples):
            with torch.cuda.stream(streams[index]):
                single_loss, losses, _ = calculate_loss(
                    params=params,
                    sample=sample,
                    variables=variables,
                    renderer=renderers[index],
                    gamma=gamma,
                    point_radius=point_radius,
                    phase=phase,
                    enable_geometry_losses=enable_geometry_losses,
                    config=config,
                )

                results[index] = (single_loss, losses)

    torch.cuda.synchronize()

    total_loss = torch.tensor(0.0, device=device)

    for result in results:
        if result is None:
            continue

        single_loss, losses = result
        total_loss = total_loss + single_loss

        for name in component_sum:
            component_sum[name] = (
                component_sum[name] + losses[name]
            )

    return total_loss, component_sum


def clone_params(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Clone all trainable tensors.
    """
    cloned_params = {}

    for name, value in params.items():
        if isinstance(value, torch.Tensor):
            cloned_params[name] = (
                value.detach()
                .clone()
                .requires_grad_(True)
            )
        else:
            cloned_params[name] = value

    return cloned_params


def build_renderer_pool(
    point_count: int,
    pool_size: int,
) -> Tuple[
    List[PulsarRenderer],
    List[torch.cuda.Stream],
]:
    """
    Create reusable Pulsar renderers and CUDA streams.
    """
    renderers = [
        PulsarRenderer(
            1024,
            1024,
            point_count,
            right_handed_system=True,
            n_track=100,
        ).cuda()
        for _ in range(pool_size)
    ]

    streams = [
        torch.cuda.Stream()
        for _ in range(pool_size)
    ]

    return renderers, streams


def set_phase1_learning_rates(
    optimizer: torch.optim.Optimizer,
    position_learning_rate: float,
) -> None:
    """
    Enable position training and freeze color training.
    """
    for group in optimizer.param_groups:
        parameter_name = group.get("name")

        if parameter_name == "means3D":
            group["lr"] = position_learning_rate
        elif parameter_name == "rgb_colors":
            group["lr"] = 0.0
        else:
            group["lr"] = 0.0


def set_phase2_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Enable color training and freeze all other parameters.
    """
    for group in optimizer.param_groups:
        if group.get("name") == "rgb_colors":
            group["lr"] = 0.025
        else:
            group["lr"] = 0.0


def get_trainable_tensors(
    params: Dict[str, Any],
) -> List[torch.Tensor]:
    """
    Return tensors that currently require gradients.
    """
    return [
        value
        for value in params.values()
        if isinstance(value, torch.Tensor)
        and value.requires_grad
    ]


def train(
    data_path: str,
    experiment_name: str,
    config: SequenceConfig,
    setting_module: ModuleType,
) -> None:
    """
    Train one Pulsar experiment.
    """
    os.makedirs(
        os.path.join("output", experiment_name),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join("output_images", experiment_name),
        exist_ok=True,
    )

    frames = load_cam_json(data_path)

    if len(frames) == 0:
        raise RuntimeError("No frames were found in cam.json.")

    scaler = GradScaler(enabled=config.amp_enabled)

    loss_log_path = os.path.join(
        "output",
        experiment_name,
        "loss_log.csv",
    )

    with open(
        loss_log_path,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(
            [
                "timestep",
                "phase",
                "iteration",
                "loss_total",
                "loss_im",
                "loss_rigid",
                "loss_iso",
                "loss_col",
                "rgb_grad_norm",
                "rgb_delta",
            ]
        )

        params, variables = setting_module.initialize_params(
            data_path,
            config.radius_phase1,
        )

        point_count = int(params["means3D"].shape[0])

        renderers, cuda_streams = build_renderer_pool(
            point_count=point_count,
            pool_size=1,
        )

        debug_renderer = renderers[0]
        output_params = []

        try:
            for timestep, frame_record in enumerate(frames):
                real_frame_id = frame_record.get(
                    "_real_frame_id"
                )

                if real_frame_id is None:
                    tqdm.write(
                        f"[Frame] Timestep {timestep}"
                    )
                else:
                    tqdm.write(
                        f"[Frame] Timestep {timestep}: "
                        f"frame{real_frame_id}"
                    )

                phase1_dataset = get_phase_dataset(
                    data_path=data_path,
                    frame_record=frame_record,
                    phase="move",
                    camera_distance=config.camera_distance,
                )

                phase2_dataset = get_phase_dataset(
                    data_path=data_path,
                    frame_record=frame_record,
                    phase="refine",
                    camera_distance=config.camera_distance,
                )

                if len(phase1_dataset) == 0:
                    raise RuntimeError(
                        f"No phase-1 samples were found "
                        f"for timestep {timestep}."
                    )

                if len(phase2_dataset) == 0:
                    raise RuntimeError(
                        f"No phase-2 samples were found "
                        f"for timestep {timestep}."
                    )

                fixed_phase1_sample = pick_fixed_debug_sample(
                    dataset=phase1_dataset,
                    camera_distance=config.camera_distance,
                )

                fixed_phase2_sample = pick_fixed_debug_sample(
                    dataset=phase2_dataset,
                    camera_distance=config.camera_distance,
                )

                if timestep == 0:
                    output_params.append(
                        params2cpu(params, True)
                    )

                    variables = (
                        setting_module.initialize_post_first_timestep(
                            params,
                            variables,
                        )
                    )
                    continue

                params, variables = (
                    setting_module.initialize_per_timestep(
                        params,
                        variables,
                        velocity_init=False,
                    )
                )

                optimizer = setting_module.initialize_optimizer(
                    params
                )
                scheduler = setting_module.initialize_scheduler(
                    optimizer
                )

                set_phase1_learning_rates(
                    optimizer=optimizer,
                    position_learning_rate=(
                        config.position_learning_rate
                    ),
                )

                best_loss = float("inf")
                best_params = None

                phase1_progress = tqdm(
                    range(1600),
                    desc=(
                        f"timestep {timestep} | "
                        f"phase 1 (move)"
                    ),
                )

                for iteration in phase1_progress:
                    batch = build_training_batch(
                        phase1_dataset,
                        batch_size=5,
                    )

                    optimizer.zero_grad(set_to_none=True)

                    total_loss_sum = torch.tensor(
                        0.0,
                        device=params["means3D"].device,
                    )

                    component_sum = {
                        "im": torch.tensor(
                            0.0,
                            device=params["means3D"].device,
                        ),
                        "rigid": torch.tensor(
                            0.0,
                            device=params["means3D"].device,
                        ),
                        "iso": torch.tensor(
                            0.0,
                            device=params["means3D"].device,
                        ),
                        "soft_col_cons": torch.tensor(
                            0.0,
                            device=params["means3D"].device,
                        ),
                    }

                    waves = [
                        batch[index : index + 1]
                        for index in range(0, len(batch), 1)
                    ]

                    for wave in waves:
                        wave_loss, wave_components = (
                            run_parallel_wave(
                                params=params,
                                samples=wave,
                                variables=variables,
                                renderers=renderers,
                                streams=cuda_streams,
                                gamma=config.phase1_gamma,
                                point_radius=(
                                    config.radius_phase1
                                ),
                                phase="move",
                                enable_geometry_losses=True,
                                config=config,
                            )
                        )

                        total_loss_sum = (
                            total_loss_sum + wave_loss
                        )

                        for name in component_sum:
                            component_sum[name] = (
                                component_sum[name]
                                + wave_components[name]
                            )

                    denominator = max(len(batch), 1)
                    total_loss = total_loss_sum / denominator

                    component_average = {
                        name: value / denominator
                        for name, value in component_sum.items()
                    }

                    with torch.no_grad():
                        old_positions = (
                            params["means3D"]
                            .detach()
                            .clone()
                        )

                    scaler.scale(total_loss).backward()

                    try:
                        torch.nn.utils.clip_grad_norm_(
                            get_trainable_tensors(params),
                            max_norm=2.0,
                        )
                    except RuntimeError:
                        pass

                    scaler.step(optimizer)
                    scaler.update()

                    with torch.no_grad():
                        movement = (
                            params["means3D"].detach()
                            - old_positions
                        ).norm().item()

                    scheduler.step(
                        total_loss.detach().float().item()
                    )

                    current_loss = float(
                        total_loss.detach().float()
                    )

                    if current_loss < best_loss:
                        best_loss = current_loss

                        if best_params is not None:
                            del best_params

                        best_params = clone_params(params)

                    if iteration % 10 == 0:
                        csv_writer.writerow(
                            [
                                timestep,
                                "move",
                                iteration,
                                current_loss,
                                float(
                                    component_average["im"]
                                    .detach()
                                    .float()
                                ),
                                float(
                                    component_average["rigid"]
                                    .detach()
                                    .float()
                                ),
                                float(
                                    component_average["iso"]
                                    .detach()
                                    .float()
                                ),
                                float(
                                    component_average[
                                        "soft_col_cons"
                                    ]
                                    .detach()
                                    .float()
                                ),
                                0.0,
                                0.0,
                            ]
                        )

                        csv_file.flush()

                        phase1_progress.set_postfix(
                            {
                                "loss": f"{current_loss:.4f}",
                                "move": f"{movement:.4e}",
                                "gamma": (
                                    f"{config.phase1_gamma:.3g}"
                                ),
                            }
                        )

                    if (
                        (iteration + 1) % 1000 == 0
                        and iteration < 1599
                    ):
                        variables = (
                            setting_module
                            .initialize_post_first_timestep(
                                params,
                                variables,
                            )
                        )

                        with torch.no_grad():
                            current_points = (
                                params["means3D"].detach()
                            )

                            if "neighbor_indices" in variables:
                                variables["prev_offset"] = (
                                    current_points[
                                        variables[
                                            "neighbor_indices"
                                        ]
                                    ]
                                    - current_points[:, None]
                                ).detach()
                            else:
                                variables["prev_offset"] = (
                                    torch.zeros(
                                        (
                                            current_points.shape[0],
                                            1,
                                            3,
                                        ),
                                        device=(
                                            current_points.device
                                        ),
                                    )
                                )

                    if (
                        fixed_phase1_sample is not None
                        and (
                            iteration % 50 == 0
                            or iteration == 1599
                        )
                    ):
                        image_output_directory = os.path.join(
                            "output_images",
                            experiment_name,
                            f"timestep_{timestep:04d}",
                        )

                        os.makedirs(
                            image_output_directory,
                            exist_ok=True,
                        )

                        save_comparison_image(
                            params=params,
                            variables=variables,
                            sample=fixed_phase1_sample,
                            renderer=debug_renderer,
                            output_path=os.path.join(
                                image_output_directory,
                                (
                                    f"compare_t{timestep}_"
                                    f"phase1_i{iteration}.png"
                                ),
                            ),
                            gamma=config.phase1_gamma,
                            point_radius=(
                                config.radius_phase1
                            ),
                            config=config,
                        )

                if best_params is not None:
                    params = best_params

                try:
                    torch.save(
                        params2cpu(params, False),
                        os.path.join(
                            "output",
                            experiment_name,
                            (
                                f"t{timestep:04d}_"
                                f"best_phase1.pt"
                            ),
                        ),
                    )
                except (OSError, RuntimeError) as error:
                    tqdm.write(
                        "[Warning] Failed to save the "
                        f"phase-1 checkpoint: {error}"
                    )

                variables = (
                    setting_module.initialize_post_first_timestep(
                        params,
                        variables,
                    )
                )

                optimizer = setting_module.initialize_optimizer(
                    params
                )
                scheduler = setting_module.initialize_scheduler(
                    optimizer
                )

                set_phase2_learning_rates(optimizer)

                with torch.no_grad():
                    starting_colors = (
                        params["rgb_colors"]
                        .detach()
                        .clone()
                    )

                phase2_progress = tqdm(
                    range(400),
                    desc=(
                        f"timestep {timestep} | "
                        f"phase 2 (refine)"
                    ),
                )

                for iteration in phase2_progress:
                    batch = build_refinement_batch(
                        phase2_dataset,
                        batch_size=5,
                    )

                    optimizer.zero_grad(set_to_none=True)

                    total_loss_sum = torch.tensor(
                        0.0,
                        device=params["means3D"].device,
                    )

                    component_sum = {
                        "im": torch.tensor(
                            0.0,
                            device=params["means3D"].device,
                        ),
                        "rigid": torch.tensor(
                            0.0,
                            device=params["means3D"].device,
                        ),
                        "iso": torch.tensor(
                            0.0,
                            device=params["means3D"].device,
                        ),
                        "soft_col_cons": torch.tensor(
                            0.0,
                            device=params["means3D"].device,
                        ),
                    }

                    waves = [
                        batch[index : index + 1]
                        for index in range(0, len(batch), 1)
                    ]

                    for wave in waves:
                        wave_loss, wave_components = (
                            run_parallel_wave(
                                params=params,
                                samples=wave,
                                variables=variables,
                                renderers=renderers,
                                streams=cuda_streams,
                                gamma=config.phase2_gamma,
                                point_radius=(
                                    config.radius_phase2
                                ),
                                phase="refine",
                                enable_geometry_losses=False,
                                config=config,
                            )
                        )

                        total_loss_sum = (
                            total_loss_sum + wave_loss
                        )

                        for name in component_sum:
                            component_sum[name] = (
                                component_sum[name]
                                + wave_components[name]
                            )

                    denominator = max(len(batch), 1)
                    total_loss = total_loss_sum / denominator

                    component_average = {
                        name: value / denominator
                        for name, value in component_sum.items()
                    }

                    scaler.scale(total_loss).backward()

                    rgb_gradient_norm = 0.0

                    if params["rgb_colors"].grad is not None:
                        rgb_gradient_norm = float(
                            params["rgb_colors"]
                            .grad
                            .detach()
                            .norm()
                            .item()
                        )

                    try:
                        torch.nn.utils.clip_grad_norm_(
                            get_trainable_tensors(params),
                            max_norm=2.0,
                        )
                    except RuntimeError:
                        pass

                    scaler.step(optimizer)
                    scaler.update()

                    scheduler.step(
                        total_loss.detach().float().item()
                    )

                    with torch.no_grad():
                        rgb_delta = float(
                            (
                                params["rgb_colors"].detach()
                                - starting_colors
                            )
                            .abs()
                            .mean()
                            .item()
                        )

                    current_loss = float(
                        total_loss.detach().float()
                    )

                    if iteration % 10 == 0:
                        csv_writer.writerow(
                            [
                                timestep,
                                "refine",
                                iteration,
                                current_loss,
                                float(
                                    component_average["im"]
                                    .detach()
                                    .float()
                                ),
                                float(
                                    component_average["rigid"]
                                    .detach()
                                    .float()
                                ),
                                float(
                                    component_average["iso"]
                                    .detach()
                                    .float()
                                ),
                                float(
                                    component_average[
                                        "soft_col_cons"
                                    ]
                                    .detach()
                                    .float()
                                ),
                                rgb_gradient_norm,
                                rgb_delta,
                            ]
                        )

                        csv_file.flush()

                        phase2_progress.set_postfix(
                            {
                                "loss": f"{current_loss:.4f}",
                                "gamma": (
                                    f"{config.phase2_gamma:.3g}"
                                ),
                                "rgb_g": (
                                    f"{rgb_gradient_norm:.3e}"
                                ),
                                "rgb_d": f"{rgb_delta:.3e}",
                            }
                        )

                    if (
                        fixed_phase2_sample is not None
                        and (
                            iteration % 50 == 0
                            or iteration == 399
                        )
                    ):
                        image_output_directory = os.path.join(
                            "output_images",
                            experiment_name,
                            f"timestep_{timestep:04d}",
                        )

                        os.makedirs(
                            image_output_directory,
                            exist_ok=True,
                        )

                        save_comparison_image(
                            params=params,
                            variables=variables,
                            sample=fixed_phase2_sample,
                            renderer=debug_renderer,
                            output_path=os.path.join(
                                image_output_directory,
                                (
                                    f"compare_t{timestep}_"
                                    f"phase2_i{iteration}.png"
                                ),
                            ),
                            gamma=config.phase2_gamma,
                            point_radius=(
                                config.radius_phase2
                            ),
                            config=config,
                        )

                variables = (
                    setting_module.initialize_post_first_timestep(
                        params,
                        variables,
                    )
                )

                output_params.append(
                    params2cpu(params, False)
                )

                torch.cuda.empty_cache()

            save_params(output_params, experiment_name)

        finally:
            try:
                del debug_renderer
                del renderers
                del cuda_streams
            except UnboundLocalError:
                pass

            torch.cuda.empty_cache()


def gamma_tag(value: float) -> str:
    """
    Convert a gamma value into a filename-safe tag.
    """
    value_string = f"{value:.6g}"
    value_string = (
        value_string
        .replace(".", "p")
        .replace("-", "m")
    )

    return f"g{value_string}"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        required=True,
        help="Path to one training case.",
    )

    parser.add_argument(
        "--exp_name",
        "-e",
        type=str,
        required=True,
        help="Base experiment name.",
    )

    parser.add_argument(
        "--radius1",
        "-r1",
        type=float,
        default=None,
        help="Input point radius for phase 1.",
    )

    parser.add_argument(
        "--radius2",
        "-r2",
        type=float,
        default=None,
        help="Input point radius for phase 2.",
    )

    args = parser.parse_args()

    sequence = detect_sequence(
        data_path=args.data_path,
        exp_name=args.exp_name,
    )

    config = build_sequence_config(
        sequence=sequence,
        radius1=args.radius1,
        radius2=args.radius2,
    )

    setting_module = load_setting_module(
        config.setting_module
    )

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except RuntimeError:
        pass

    experiment_name = (
        f"{args.exp_name}_"
        f"{gamma_tag(config.phase1_gamma)}"
    )

    print(f"Detected sequence: {config.sequence}")
    print(f"Setting module: {config.setting_module}")
    print(f"Camera distance: {config.camera_distance}")
    print(
        "Effective radius: "
        f"{config.radius_phase1}, "
        f"{config.radius_phase2}"
    )
    print(
        "Gamma: "
        f"{config.phase1_gamma}, "
        f"{config.phase2_gamma}"
    )
    print(f"Experiment name: {experiment_name}")

    train(
        data_path=args.data_path,
        experiment_name=experiment_name,
        config=config,
        setting_module=setting_module,
    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()