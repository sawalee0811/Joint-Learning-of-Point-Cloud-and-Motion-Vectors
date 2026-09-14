# DPC

This folder contains the implementation of **Direct Point Cloud (DPC)** used in our paper:

**Joint Learning of Point Clouds and Motion Vectors for Volumetric Video: A New Paradigm and Its Downstream Applications**

DPC directly optimizes dynamic point clouds using differentiable point cloud renderers.

## Folder Structure

A training case is expected to follow a structure similar to:

```text
data_longdress6/
└── a_longdress6/
    ├── Ply/
    │   ├── frame1.ply
    │   ├── frame2.ply
    │   └── ...
    ├── cam.json
    ├── ims_1/
    └── ims_2/
```

The data preparation scripts generate the rendered training views and camera metadata required by `train.py`.

## 1. Prepare Training Data

Use the corresponding data preparation script for real or synthetic sequences.

### Real sequences

Run:

```bash
python data_prepare_real.py \
    --data_base path/to/your/data
```

The default example is configured for:

```text
a_longdress6
```

The script reads the PLY files under:

```text
path/to/your/data/data_longdress6/a_longdress6/Ply/
```

and generates:

```text
cam.json
ims_1/
ims_2/
```

The real-sequence configuration uses the coordinate scale and camera settings used in our experiments.

### Synthetic sequences

Run:

```bash
python data_prepare_synthetic.py \
    --data_base path/to/your/data
```

The script processes synthetic sequences such as `man` and `woman` and generates the same output structure:

```text
cam.json
ims_1/
ims_2/
```

The synthetic-sequence configuration uses its corresponding coordinate scale and camera settings.

## 2. Run DPC Training

After the rendered views and `cam.json` have been generated, run:

```bash
python train.py \
    --data_path path/to/your/data/data_longdress6/a_longdress6 \
    --exp_name dpc_longdress
```

Optional point-radius settings can also be specified:

```bash
python train.py \
    --data_path path/to/your/data/data_longdress6/a_longdress6 \
    --exp_name dpc_longdress \
    --radius1 2 \
    --radius2 2
```

The training script automatically detects the sequence category from the data path or experiment name and loads the corresponding real or synthetic configuration.

The learned parameters are saved to the output directory used by the training code.

## 3. Convert Learned Results to PLY

After training, use the repository-level `convert2ply.py` script.

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

- DPC uses PyTorch3D and Pulsar as differentiable point cloud renderers.
- `data_prepare_real.py` and `data_prepare_synthetic.py` already generate both the rendered views and the camera metadata required for training.
- The training code uses `ims_1` for the movement optimization phase and `ims_2` for the color refinement phase.
- The required Python dependencies should be installed according to the repository-level README.
- The default settings follow the configurations used in the paper.
