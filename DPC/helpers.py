import torch
import os
import open3d as o3d
import numpy as np
import torch.nn.functional as F

def params2rendervar(params):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def params2cpu(params, is_initial_timestep):
    if is_initial_timestep:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if k in ['means3D', 'rgb_colors', 'unnorm_rotations']}
    return res

def detect_bg_rgb(np_img_uint8: np.ndarray) -> tuple:
    """Detect background color (black or white) by sampling image corners.
    Return uint8 tuple (0,0,0) or (255,255,255)."""
    if np_img_uint8.ndim == 2:
        img = np.repeat(np_img_uint8[..., None], 3, axis=2)
    else:
        img = np_img_uint8

    h, w, _ = img.shape
    s = max(1, min(h, w) // 20)  # corner patch size ~5% of min side
    patches = [
        img[0:s, 0:s],              # top-left
        img[0:s, w-s:w],            # top-right
        img[h-s:h, 0:s],            # bottom-left
        img[h-s:h, w-s:w],          # bottom-right
    ]
    corner_mean = np.mean([p.reshape(-1, 3).mean(axis=0) for p in patches], axis=0)
    # Decide by luminance threshold
    lum = 0.299*corner_mean[0] + 0.587*corner_mean[1] + 0.114*corner_mean[2]
    return (0, 0, 0) if lum < 128.0 else (255, 255, 255)


def save_params(output_params, exp):
    import numpy as np
    import os

    to_save = {}
    for k in output_params[0].keys():
        try:
            if k in output_params[1].keys():
                stacked = []
                ref_shape = None
                for idx, params in enumerate(output_params):
                    arr = params[k]
                    if isinstance(arr, torch.Tensor):
                        arr = arr.detach().cpu().numpy()
                    arr = np.asarray(arr).astype(np.float32)  # 強制轉換乾淨 array
                    if idx == 0:
                        ref_shape = arr.shape
                    elif arr.shape != ref_shape:
                        print(f"[WARNING] Shape mismatch at key={k}, index={idx}: {arr.shape} != {ref_shape}")
                    stacked.append(arr)

                stacked_np = np.stack(stacked)
                to_save[k] = stacked_np

            else:
                arr = output_params[0][k]
                if isinstance(arr, torch.Tensor):
                    arr = arr.detach().cpu().numpy()
                arr = np.asarray(arr).astype(np.float32)  # 強制轉換乾淨 array
                to_save[k] = arr

        except Exception as e:
            print(f"[FATAL] Failed to process key: {k}")
            print(f"        Reason: {e}")
            raise e

    os.makedirs(f"./output/{exp}", exist_ok=True)
    np.savez(f"./output/{exp}/params", **to_save)

def sobel_grad_map(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 3 and img.shape[-1] == 3:
        gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).unsqueeze(0).unsqueeze(0)
    else:
        gray = img.unsqueeze(0).unsqueeze(0)
    ky = torch.tensor([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
    kx = torch.tensor([[-1,  0,  1],
                    [-2,  0,  2],
                    [-1,  0,  1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag[0, 0]  # (H, W)

def make_edge_weight(gt_img: torch.Tensor, alpha: float = 3.0, beta: float = 0.5) -> torch.Tensor:
    """
    Edge-aware weight map from GT gradient.
    weight = 1 + alpha * (normalized_grad ** beta). Output shape: (H, W, 1)
    """
    gmag = sobel_grad_map(gt_img)
    gmag = gmag / (gmag.max() + 1e-6)
    w = 1.0 + alpha * (gmag ** beta)
    return w.unsqueeze(-1)