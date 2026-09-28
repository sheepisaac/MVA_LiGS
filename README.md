# MVA-LiGS

MVA-LiGS initializes LiDAR-anchored 3D Gaussians with robust multi-view color
attributes and can regularize Gaussian color during training. This repository
is dataset-agnostic: it expects a standard COLMAP reconstruction and a LiDAR
point cloud already expressed in the same coordinate frame and scale.

The code is based on the official
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
implementation pinned at commit
`472689c0dc70417448fb451bf529ae532d32c095`.

## Repository layout

- `prepare_mva_ligs_dataset.py`: projects LiDAR points into calibrated views,
  aggregates multi-view color, and creates a 3DGS-compatible dataset.
- `train_mva_ligs.py`: trains 3DGS with the optional MVA color prior loss.
- `gaussian_model_mva.py`: Gaussian model that keeps MVA priors synchronized
  through densification and pruning.
- `gaussian_renderer_mva.py`: compatibility renderer for the official and
  depth-enabled rasterizer variants.
- `render_mva_ligs.py`: compatible train/test rendering entry point.
- `register_lidar_to_colmap.py`: optional rigid LiDAR-to-SfM registration for
  point clouds that already have compatible scale and units.
- `tools/filter_lidar_outliers.py`: optional LiDAR outlier filtering.
- `tools/evaluate.py`: rendering and metric helper.
- `scripts/run_pipeline.sh`: portable preparation and training entry point.

ETH3D-specific preprocessing and experiment scripts are intentionally not part
of the repository.

## Input format

The COLMAP source directory must have this structure:

```text
scene/
├── images/
└── sparse/0/
    ├── cameras.bin
    └── images.bin
```

The LiDAR input must be a PLY point cloud in the same coordinate frame and scale
as the COLMAP reconstruction. `points3D.bin` is not required because MVA-LiGS
uses the LiDAR points as Gaussian anchors.

## Installation

Clone recursively so the pinned 3DGS code and its CUDA extensions are present:

```bash
git clone --recursive <repository-url>
cd MVA_LiGS
```

Install a CUDA-compatible PyTorch build first. Then install the remaining
Python packages and the two 3DGS CUDA extensions:

```bash
python -m pip install -r requirements.txt
python -m pip install third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
python -m pip install third_party/gaussian-splatting/submodules/simple-knn
```

COLMAP is only needed if the input images have not already been reconstructed.

## Run

```bash
scripts/run_pipeline.sh \
  /path/to/scene \
  /path/to/aligned_lidar.ply \
  /path/to/work_dir
```

By default all registered views are used, up to 250,000 LiDAR points are kept,
and training runs for 30,000 iterations. Settings can be changed through
environment variables:

```bash
MVA_MODE=zbuffer_bilinear \
MAX_POINTS=500000 \
MAX_COLOR_VIEWS=100 \
ITERATIONS=30000 \
CUDA_VISIBLE_DEVICES=0 \
scripts/run_pipeline.sh /path/to/scene /path/to/aligned_lidar.ply /path/to/work
```

Additional 3DGS training arguments may be appended to the command. They are
passed directly to `train_mva_ligs.py`.

## Optional preprocessing

Filter an already aligned LiDAR cloud:

```bash
python tools/filter_lidar_outliers.py \
  --input_ply input.ply \
  --output_ply filtered.ply \
  --report_json filter_report.json
```

For rigid registration against a COLMAP point-cloud export:

```bash
python register_lidar_to_colmap.py \
  --lidar_input lidar.ply \
  --sfm_points points3D.ply \
  --out_dir registration
```

This registration does not estimate scale. For arbitrary-scale SfM data,
recover the scale before running it or provide a pre-aligned LiDAR PLY.

## Evaluation

```bash
python tools/evaluate.py --model /path/to/work_dir/output --iteration -1
```

Use `--help` on each Python script for all available options.

## License

The training and Gaussian model code derives from Inria/MPII Gaussian
Splatting and is distributed under its non-commercial research and evaluation
license. See `LICENSE.md` and `NOTICE.md`.
