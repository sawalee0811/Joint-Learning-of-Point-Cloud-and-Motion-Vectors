# Joint-Learning-of-Point-Cloud-and-Motion-Vectors
This is the source code for the paper: Joint Learning of Point Clouds and Motion Vectors for Volumetric Video: A New Paradigm and Its Downstream Applications.

## Dataset
We use the [8i dataset](https://plenodb.jpeg.org/pc/8ilabs) amd [MMsys dataset](https://dl.acm.org/doi/abs/10.1145/3587819.3592546)

## Environment
```Bash
git clone git@github.com:sawalee0811/Joint-Learning-of-Point-Cloud-and-Motion-Vectors.git
conda create -n JLPM python=3.9 -y
conda activate JLPM

pip install "numpy>1.26,<2" "pandas>=2.2.3" "open3d>0.16" scikit-learn tqdm "pillow>=11.0"
```

### Gaussian as a Proxy (GaaP) 
GaaP algorithm is our first algorithm that leverage Dynamic 3DGS for training. Please follow their repo [here](https://github.com/JonathonLuiten/Dynamic3DGaussians) to setup the environment.

### Directly Point Cloud (DPC)
For directly using point cloud to train, We use the Pytorch3D library for training. Please install the Pytorch3D [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

## Run
GaaP:
```Bash
python gaap/train.py
```

DPC:
```Bash
python render_gt.py
python dpc/train.py
```
