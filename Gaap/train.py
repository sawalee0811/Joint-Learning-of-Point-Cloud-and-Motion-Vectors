import torch
import os
import json
import torchvision.utils
import copy
import argparse
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from pathlib import Path

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import (
    setup_camera,
    l1_loss_v1,
    l1_loss_v2,
    weighted_l2_loss_v1,
    weighted_l2_loss_v2,
    quat_mult,
    o3d_knn,
    params2rendervar,
    params2cpu,
    save_params,
)
from external import calc_ssim, calc_psnr


def get_base_seq_from_seq(seq):
    """
    Convert a sequence name such as a_redandblack6 to redandblack6.
    """
    parts = seq.split("_", 1)

    if len(parts) != 2:
        raise ValueError(
            f"Invalid sequence name: {seq}. Expected format like a_redandblack6."
        )

    return parts[1]

def resolve_dataset_root(seq, data_root):
    """
    Resolve dataset root.

    Example:
        seq = a_redandblack6
        data_root = path/to/your/data

    Resolved path:
        path/to/your/data/data_redandblack6/a_redandblack6
    """
    base_seq = get_base_seq_from_seq(seq)
    dataset_root = Path(data_root) / f"data_{base_seq}" / seq

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    return dataset_root

def resolve_train_meta_path(dataset_root):
    """
    Resolve train_meta.json path.

    The expected path is:
        dataset_root/train_meta.json

    A fallback path is also supported:
        dataset_root/Ply/train_meta.json
    """
    candidates = [
        dataset_root / "train_meta.json",
        dataset_root / "Ply" / "train_meta.json",
    ]

    for path in candidates:
        if path.is_file():
            return path

    checked = "\n".join([f"  - {path}" for path in candidates])
    raise FileNotFoundError(f"Training metadata file not found. Checked paths:\n{checked}")

def get_dataset(t, md, dataset_root):
    dataset = []

    for c in range(len(md["fn"][t])):
        w = md["w"]
        h = md["h"]
        k = md["k"][t][c]
        w2c = md["w2c"][t][c]

        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)

        fn = md["fn"][t][c].replace("\\", "/")
        seg_fn = str(Path(fn).with_suffix(".png")).replace("\\", "/")

        im_path = dataset_root / "ims" / fn
        seg_path = dataset_root / "seg" / seg_fn

        if not im_path.is_file():
            raise FileNotFoundError(f"Image file not found: {im_path}")

        if not seg_path.is_file():
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

        im = np.array(copy.deepcopy(Image.open(im_path).convert("RGB")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255

        seg = np.array(copy.deepcopy(Image.open(seg_path))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))

        dataset.append({"cam": cam, "im": im, "seg": seg_col, "id": c})

    return dataset

def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()

    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data

def initialize_params(dataset_root, md, scale):
    init_pt_cld_path = dataset_root / "init_pt_cld.npz"

    if not init_pt_cld_path.is_file():
        raise FileNotFoundError(f"Initial point cloud file not found: {init_pt_cld_path}")

    init_pt_cld = np.load(init_pt_cld_path)["data"]

    seg = init_pt_cld[:, 6]
    max_cams = 100
    mean_dist = np.full(init_pt_cld.shape[0], scale)

    params = {
        "means3D": init_pt_cld[:, :3],
        "rgb_colors": init_pt_cld[:, 3:6],
        "seg_colors": np.stack((seg, np.ones_like(seg), np.zeros_like(seg)), -1),
        "unnorm_rotations": np.tile([0, 0, 0, 1], (seg.shape[0], 1)),
        "logit_opacities": np.ones((seg.shape[0], 1)),
        "log_scales": np.tile(mean_dist[..., None], (1, 3)),
        "cam_m": np.zeros((max_cams, 3)),
        "cam_c": np.zeros((max_cams, 3)),
    }

    params = {
        k: torch.nn.Parameter(
            torch.tensor(v).cuda().float().contiguous().requires_grad_(True)
        )
        for k, v in params.items()
    }

    cam_centers = np.linalg.inv(md["w2c"][0])[:, :3, 3]
    scene_radius = 1.1 * np.max(
        np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1)
    )

    variables = {
        "max_2D_radius": torch.zeros(params["means3D"].shape[0]).cuda().float(),
        "scene_radius": scene_radius,
        "means2D_gradient_accum": torch.zeros(params["means3D"].shape[0]).cuda().float(),
        "denom": torch.zeros(params["means3D"].shape[0]).cuda().float(),
    }

    return params, variables

def initialize_optimizer(params, variables):
    lrs = {
        "means3D": 0.00016 * variables["scene_radius"],
        "rgb_colors": 0.0025,
        "seg_colors": 0.0,
        "unnorm_rotations": 0.0,
        "logit_opacities": 0.0,
        "log_scales": 0.0,
        "cam_m": 1e-4,
        "cam_c": 1e-4,
    }

    param_groups = [
        {"params": [v], "name": k, "lr": lrs[k]} for k, v in params.items()
    ]

    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def get_loss(params, curr_data, variables, is_initial_timestep):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar["means2D"].retain_grad()

    raster_settings = curr_data["cam"]
    im, radius, _ = Renderer(raster_settings=raster_settings)(**rendervar)

    curr_id = curr_data["id"]
    im = torch.exp(params["cam_m"][curr_id])[:, None, None] * im + params["cam_c"][curr_id][:, None, None]

    import imageio.v2 as imageio

    debug_dir = Path("./debug_images")
    debug_dir.mkdir(exist_ok=True)

    def tensor_to_uint8(t):
        t = t.detach().cpu().clamp(0, 1)

        if t.ndim == 3 and t.shape[0] in [1, 3]:
            t = t.permute(1, 2, 0)

        return (t.numpy() * 255).astype(np.uint8)

    im_save = tensor_to_uint8(im)
    gt_save = tensor_to_uint8(curr_data["im"])

    imageio.imwrite(debug_dir / f"im_render_{curr_id}.png", im_save)
    imageio.imwrite(debug_dir / f"im_gt_{curr_id}.png", gt_save)

    losses["im"] = 0.8 * l1_loss_v1(im, curr_data["im"]) + 0.2 * (
        1.0 - calc_ssim(im, curr_data["im"])
    )

    variables["means2D"] = rendervar["means2D"]

    segrendervar = params2rendervar(params)
    segraster_settings = curr_data["cam"]
    seg, _, _ = Renderer(raster_settings=segraster_settings)(**segrendervar)

    if not is_initial_timestep:
        is_fg = (params["seg_colors"][:, 0] > 0.5).detach()
        fg_pts = rendervar["means3D"][is_fg]

        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]

        losses["rigid"] = weighted_l2_loss_v2(
            curr_offset,
            variables["prev_offset"],
            variables["neighbor_weight"],
        )

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)

        losses["iso"] = weighted_l2_loss_v1(
            curr_offset_mag,
            variables["neighbor_dist"],
            variables["neighbor_weight"],
        )

        losses["floor"] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar["means3D"][~is_fg]
        losses["bg"] = l1_loss_v2(bg_pts, variables["init_bg_pts"])

        losses["soft_col_cons"] = l1_loss_v2(
            params["rgb_colors"],
            variables["prev_col"],
        )

    loss_weights = {
        "im": 1.0,
        "seg": 3.0,
        "rigid": 4.0,
        "iso": 1.0,
        "floor": 0.0,
        "bg": 0.0,
        "soft_col_cons": 0.01,
    }

    if is_initial_timestep:
        loss = torch.tensor(0.0, requires_grad=True)
    else:
        loss = sum([loss_weights[k] * v for k, v in losses.items()])

    seen = radius > 0
    variables["max_2D_radius"][seen] = torch.max(
        radius[seen],
        variables["max_2D_radius"][seen],
    )
    variables["seen"] = seen

    return loss, variables

def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group["params"][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group["params"][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group["params"][0]] = stored_state
        params[k] = group["params"][0]

    return params

def initialize_per_timestep(params, variables, optimizer):
    pts = params["means3D"]
    new_pts = pts + (pts - variables["prev_pts"])

    is_fg = params["seg_colors"][:, 0] > 0.5
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]

    variables["prev_offset"] = prev_offset.detach()
    variables["prev_col"] = params["rgb_colors"].detach()
    variables["prev_pts"] = pts.detach()

    new_params = {"means3D": new_pts}

    optimizer = ensure_optimizer_initialized(optimizer)
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables

def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params["seg_colors"][:, 0] > 0.5

    init_fg_pts = params["means3D"][is_fg]
    init_bg_pts = params["means3D"][~is_fg]

    neighbor_sq_dist, neighbor_indices = o3d_knn(
        init_fg_pts.detach().cpu().numpy(),
        num_knn,
    )

    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)

    variables["neighbor_indices"] = torch.tensor(
        neighbor_indices
    ).cuda().long().contiguous()

    variables["neighbor_weight"] = torch.tensor(
        neighbor_weight
    ).cuda().float().contiguous()

    variables["neighbor_dist"] = torch.tensor(
        neighbor_dist
    ).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["prev_pts"] = params["means3D"].detach()

    params_to_fix = ["cam_m", "cam_c"]

    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group["lr"] = 0.0

    return variables

def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _ = Renderer(raster_settings=data["cam"])(**params2rendervar(params))

        curr_id = data["id"]
        im = torch.exp(params["cam_m"][curr_id])[:, None, None] * im + params["cam_c"][curr_id][:, None, None]

        psnr = calc_psnr(im, data["im"]).mean()

        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)

def ensure_optimizer_initialized(optimizer):
    for group in optimizer.param_groups:
        for v in group["params"]:
            optimizer.state[v] = {
                "step": torch.tensor(1.0, device=v.device),
                "exp_avg": torch.zeros_like(v),
                "exp_avg_sq": torch.zeros_like(v),
            }

    return optimizer

def train(seq, exp, scale, data_root):
    output_check_path = f"./output/{exp}/{seq}"

    if os.path.exists(output_check_path):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return

    dataset_root = resolve_dataset_root(seq, data_root)
    train_meta_path = resolve_train_meta_path(dataset_root)

    print(f"[INFO] Sequence: {seq}")
    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Train metadata: {train_meta_path}")

    with open(train_meta_path, "r") as f:
        md = json.load(f)

    num_timesteps = len(md["fn"])

    params, variables = initialize_params(dataset_root, md, scale)
    optimizer = initialize_optimizer(params, variables)

    output_params = []

    for t in range(0, num_timesteps):
        dataset = get_dataset(t, md, dataset_root)
        todo_dataset = []

        is_initial_timestep = t == 0

        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)

        num_iter_per_timestep = 1 if is_initial_timestep else 2000 # 5000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")

        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)

            loss, variables = get_loss(
                params,
                curr_data,
                variables,
                is_initial_timestep,
            )

            loss.backward()

            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        progress_bar.close()

        output_params.append(params2cpu(params, is_initial_timestep))

        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)

    save_params(output_params, seq, exp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Cloud Training Script")

    parser.add_argument(
        "--base_exp_name",
        "-b",
        type=str,
        default="mm",
        help="Base name for the experiment",
    )

    parser.add_argument(
        "--sequence",
        "-s",
        type=str,
        default="a_redandblack6",
        help="Sequence name, such as a_woman6",
    )

    parser.add_argument(
        "--scale",
        "-l",
        type=float,
        default="-10",
        help="Scale factor for point cloud",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="path/to/your/data",
        help="Root folder that contains data_redandblack6, data_woman6, etc.",
    )

    args = parser.parse_args()

    base_exp_name = args.base_exp_name
    sequence = args.sequence
    scale = args.scale
    data_root = args.data_root

    exp_name = f"{base_exp_name}_{sequence}_scale{float(scale)}"

    print(
        f"Running experiment: {exp_name} "
        f"for sequence: {sequence} "
        f"with scale: {scale}"
    )

    train(
        seq=sequence,
        exp=exp_name,
        scale=scale,
        data_root=data_root,
    )

    torch.cuda.empty_cache()