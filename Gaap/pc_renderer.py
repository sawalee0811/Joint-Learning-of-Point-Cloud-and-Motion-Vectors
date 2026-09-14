import os
import json
import argparse
import numpy as np
from PIL import Image
import open3d as o3d
from tqdm import tqdm


# ---------- IO ----------

def load_point_cloud(ply_path: str) -> o3d.geometry.PointCloud:
    return o3d.io.read_point_cloud(ply_path)

def read_camera_params_from_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["k"], data["w2c"], data["cam_id"], data["fn"]


# ---------- Camera & Render ----------

def build_intrinsic(k: np.ndarray, width: int, height: int) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    )

def set_camera(renderer: o3d.visualization.rendering.OffscreenRenderer,
               intrinsic: o3d.camera.PinholeCameraIntrinsic,
               extrinsic_w2c: np.ndarray):
    renderer.setup_camera(intrinsic, extrinsic_w2c)

def render_color_png(renderer: o3d.visualization.rendering.OffscreenRenderer,
                    pcd: o3d.geometry.PointCloud,
                    intrinsic: o3d.camera.PinholeCameraIntrinsic,
                    extrinsic_w2c: np.ndarray,
                    img_path: str,
                    point_size: float):
    # material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.point_size = float(point_size)

    # add + camera
    renderer.scene.add_geometry("pcd", pcd, mat)
    set_camera(renderer, intrinsic, extrinsic_w2c)

    # render to image (RGB, background = pure white)
    color = renderer.render_to_image()

    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    # Ensure PNG to preserve exact 255 values
    o3d.io.write_image(img_path, color)

    renderer.scene.clear_geometry()


# ---------- Mask (post-process from saved file) ----------

def make_mask_from_rendered_png(img_path: str, mask_path: str):
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)  # H,W,3
    bg = (arr[..., 0] >= 230) & (arr[..., 1] >= 230) & (arr[..., 2] >= 230)
    mask = np.where(bg, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    Image.fromarray(mask).save(mask_path)


# ---------- Main ----------

def renderer_main(ply_folder: str, out_folder: str, json_path: str,
                point_size: float, width: int = 600, height: int = 600):
    ply_files = sorted([os.path.join(ply_folder, f)
                        for f in os.listdir(ply_folder) if f.lower().endswith(".ply")])

    img_dir = os.path.join(out_folder, "ims")
    seg_dir = os.path.join(out_folder, "seg")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    # Offscreen renderer with PURE WHITE background (exact 255)
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # RGBA = (255,255,255,255)

    k_list, w2c_list, cam_ids, fn_list = read_camera_params_from_json(json_path)
    n = min(len(ply_files), len(k_list), len(w2c_list), len(cam_ids), len(fn_list))

    for i in tqdm(range(n), desc="Rendering", total=n):
        pcd = load_point_cloud(ply_files[i])

        for j, _ in enumerate(cam_ids[i]):
            k = np.array(k_list[i][j], dtype=np.float64)
            w2c = np.array(w2c_list[i][j], dtype=np.float64)
            intrinsic = build_intrinsic(k, width, height)

            img_name = fn_list[i][j]
            img_path = os.path.join(img_dir, img_name)
            base, _ = os.path.splitext(img_name)
            mask_path = os.path.join(seg_dir, f"{base}.png")

            # 1) render color with white background
            render_color_png(renderer, pcd, intrinsic, w2c, img_path, point_size)
            # 2) post-process to mask with strict 255 check
            make_mask_from_rendered_png(img_path, mask_path)

    print("Render complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render point clouds then create masks from pure-white background.")
    parser.add_argument("-i", "--ply_folder", required=True, type=str, help="Folder with PLY files.")
    parser.add_argument("-o", "--output_image_folder", required=True, type=str, help="Output folder (ims/ and seg/ will be created).")
    parser.add_argument("-j", "--json_file", required=True, type=str, help="Camera meta JSON (k, w2c, cam_id, fn).")
    parser.add_argument("-s", "--size", default=10.0, type=float, help="Point size (default=10).")
    parser.add_argument("--width", default=600, type=int, help="Image width.")
    parser.add_argument("--height", default=600, type=int, help="Image height.")
    args = parser.parse_args()

    renderer_main(args.ply_folder, args.output_image_folder, args.json_file, args.size, args.width, args.height)
