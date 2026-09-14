# GaaP

This folder contains the implementation of **Gaussian as a Proxy (GaaP)** used in our paper:

**Joint Learning of Point Clouds and Motion Vectors for Volumetric Video: A New Paradigm and Its Downstream Applications**

GaaP uses Dynamic 3D Gaussian Splatting as the differentiable representation during joint learning.

## Folder Structure

A training case is expected to follow a structure similar to:

```text
Data/
└── a_longdress6/
    ├── Ply/
    │   ├── frame1.ply
    │   ├── frame2.ply
    │   └── ...
    ├── all_meta.json
    ├── train_meta.json
    ├── init_pt_cld.npz
    ├── ims/
    └── seg/
```

`all_meta.json` stores the original camera parameters and is used for rendering the input point clouds.

`train_meta.json` stores the camera parameters used by the GaaP training code.

## 1. Prepare Camera Metadata

Run `dataloader.py` to generate the camera metadata.

Example:

```bash
python dataloader.py \
    --ply_file path/to/your/data/frame1.ply \
    --output_dir path/to/your/data/a_longdress6 \
    --num_frames 6
```

The script generates:

```text
all_meta.json
train_meta.json
```

The default configuration uses 100 training cameras.

## 2. Render 2D Training Views

Run `pc_render.py` with `all_meta.json` to render the input point clouds into 2D images.

Example:

```bash
python pc_render.py \
    -i path/to/your/data/a_longdress6/Ply \
    -o path/to/your/data/a_longdress6 \
    -j path/to/your/data/a_longdress6/all_meta.json \
    -s 2
```

The script generates:

```text
ims/
seg/
```

Use the point size that matches the sequence used in your experiment.

## 3. Generate the Initial Point Cloud

Run `generate_init_point_cloud.py` to generate `init_pt_cld.npz` from the first PLY frame of each training case.

Example:

```bash
python generate_init_point_cloud.py \
    --data_root path/to/your/data/data_longdress6
```

The script automatically uses different coordinate scales for synthetic and real sequences.

## 4. Run GaaP Training

Run `train.py` after `train_meta.json`, `ims/`, `seg/`, and `init_pt_cld.npz` have been generated.

Example:

```bash
python train.py \
    --base_exp_name gaap \
    --sequence a_longdress6 \
    --scale -10 \
    --data_root path/to/your/data
```

The training code expects a directory structure such as:

```text
path/to/your/data/
└── data_longdress6/
    └── a_longdress6/
```

The learned parameters are saved to the output directory used by the training code.

## 5. Convert Learned Results to PLY

After training, use the repository-level `convert2ply.py` script to convert the learned point cloud parameters back to PLY files.

For real sequences:

```bash
python ../convert2ply.py \
    --params path/to/params.npz \
    --output_dir path/to/output/Ply \
    --scale 500
```

For synthetic sequences:

```bash
python ../convert2ply.py \
    --params path/to/params.npz \
    --output_dir path/to/output/Ply \
    --scale 100
```

## Notes

- GaaP is based on Dynamic 3D Gaussian Splatting. Please follow the original Dynamic 3D Gaussians repository for the required 3DGS environment and rasterizer setup.
- The rendering and training scripts assume that the required Python dependencies have already been installed as described in the repository-level README.
- The default experimental settings follow the settings used in the paper.
