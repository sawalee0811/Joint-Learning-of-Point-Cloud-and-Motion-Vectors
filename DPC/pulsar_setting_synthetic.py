import numpy as np
import torch
from helpers import o3d_knn

BATCH_SIZE = 6
VELOCITY_INIT = False  # True => pts += (pts - prev_pts)
WORLD_SCALE = 1.0/100.0

def load_npz(file_path, device, r):
    arr = np.load(file_path)

    # Support both .npz with a 'data' key and raw .npy arrays
    if isinstance(arr, np.lib.npyio.NpzFile):
        if 'data' in arr:
            data = arr['data']
        elif 'points' in arr:
            data = arr['points']
        else:
            raise KeyError(
                f"{file_path} does not contain 'data' or 'points' keys."
            )
    else:
        # Raw .npy
        data = arr

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Unexpected array shape {data.shape} in {file_path}; need at least 3 columns (xyz).")

    # xyz
    xyz = torch.from_numpy(data[:, 0:3]).float().to(device)

    # rgb (optional). If not provided, default to 0.5 gray.
    if data.shape[1] >= 6:
        rgb_np = data[:, 3:6]
        # If input is 0-255, normalize; otherwise assume 0-1
        if rgb_np.max() > 1.0:
            rgb_np = rgb_np / 255.0
        rgb = torch.from_numpy(rgb_np).float().to(device)
    else:
        rgb = torch.ones_like(xyz, dtype=torch.float32, device=device) * 0.5

    # radius vector (scaled for consistency within this module)
    radius = torch.full((xyz.shape[0],), float(r) * WORLD_SCALE, dtype=torch.float32, device=device)

    return xyz, rgb, radius

# ---------- Param init ----------
def get_batch(todo_dataset, dataset, batch_size=BATCH_SIZE):
    if not todo_dataset:
        todo_dataset.extend(dataset)
    batch_size = min(batch_size, len(todo_dataset))
    batch = [todo_dataset.pop(torch.randint(len(todo_dataset), (1,)).item()) for _ in range(batch_size)]
    return batch


def initialize_params(data_path, load_r):
    device = torch.device("cuda")
    xyz, rgb, radius = load_npz(f"{data_path}/init_pt_cld.npz", device, load_r)
    params = {
        "means3D": torch.nn.Parameter(xyz.clone().contiguous().requires_grad_(True)),
        "rgb_colors": torch.nn.Parameter(rgb.clone().contiguous().requires_grad_(True)),
        "seg_colors": torch.nn.Parameter(torch.stack(
            (radius, torch.ones_like(radius), torch.ones_like(radius)), -1
        ).clone().contiguous().requires_grad_(True)),
    }
    variables = {
        "max_2D_radius": torch.zeros(xyz.shape[0], device=device),
        "means2D_gradient_accum": torch.zeros(xyz.shape[0], device=device),
        "denom": torch.zeros(xyz.shape[0], device=device),
    }
    return params, variables


def initialize_optimizer(params):
    lrs = {"means3D": 0.0, "rgb_colors": 0.0, "seg_colors": 0.0}
    param_groups = [{"params": [params[k]], "name": k, "lr": lrs[k]} for k in ["means3D", "rgb_colors", "seg_colors"]]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200, verbose=True, min_lr=1e-5
    )

# ---------- Timestep init ----------
def ensure_optimizer_initialized(optimizer):
    for group in optimizer.param_groups:
        for v in group['params']:
            optimizer.state[v] = {
                "step": torch.tensor(1.0, device=v.device),
                "exp_avg": torch.zeros_like(v),
                "exp_avg_sq": torch.zeros_like(v),
            }
    return optimizer

def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        # print(stored_state)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params


def initialize_per_timestep(params, variables, velocity_init=VELOCITY_INIT):
    pts_now = params["means3D"].detach()
    if velocity_init and ("prev_pts" in variables):
        delta = pts_now - variables["prev_pts"]
        new_pts = pts_now + delta
    else:
        new_pts = pts_now

    if "neighbor_indices" in variables:
        prev_offset = pts_now[variables["neighbor_indices"]] - pts_now[:, None]
    else:
        prev_offset = torch.zeros((pts_now.shape[0], 1, 3), device=pts_now.device)

    variables["prev_offset"] = prev_offset.detach()
    variables["prev_col"] = params["rgb_colors"].detach()
    variables["prev_pts"] = pts_now

    params["means3D"] = torch.nn.Parameter(new_pts.clone().requires_grad_(True))
    return params, variables


def initialize_post_first_timestep(params, variables, num_knn=20):
    """
    Build KNN-based regularizers. Since xyz was scaled by WORLD_SCALE,
    we adjust the exponential weight's coefficient to keep the same
    effective falloff as in the unscaled space.
    """
    fg_pts = params["means3D"]
    neighbor_sq_dist, neighbor_indices = o3d_knn(fg_pts.detach().cpu().numpy(), num_knn)

    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)

    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["prev_pts"] = params['means3D'].detach()
    return variables