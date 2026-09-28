#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_pipeline.sh SOURCE_COLMAP ALIGNED_LIDAR_PLY WORK_DIR [TRAIN_ARGS...]

SOURCE_COLMAP must contain images/ and sparse/0/{cameras.bin,images.bin}.
The LiDAR PLY must already be in the same coordinate frame and scale as COLMAP.

Optional environment variables:
  PYTHON, GS_ROOT, CUDA_VISIBLE_DEVICES
  MVA_MODE, MAX_POINTS, MAX_COLOR_VIEWS, VOXEL_SIZE, PATCH_RADIUS
  ITERATIONS, RESOLUTION, DATA_DEVICE
  LAMBDA_MVA_COLOR, MVA_COLOR_MIN_VIEWS, MVA_COLOR_MAX_RGB_STD
  MVA_COLOR_CONFIDENCE_POWER, MVA_COLOR_DECAY_START_ITER
  MVA_COLOR_DECAY_UNTIL_ITER, MVA_ANCHOR_MAX_POINTS
EOF
}

if [[ $# -lt 3 ]]; then
  usage >&2
  exit 2
fi

SOURCE_COLMAP="$1"
LIDAR_PLY="$2"
WORK_DIR="$3"
shift 3

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GS_ROOT="${GS_ROOT:-${REPO_ROOT}/third_party/gaussian-splatting}"
PYTHON="${PYTHON:-python}"
export GS_ROOT

if [[ ! -d "${SOURCE_COLMAP}/images" ]]; then
  echo "Missing image directory: ${SOURCE_COLMAP}/images" >&2
  exit 1
fi
for file in cameras.bin images.bin; do
  if [[ ! -f "${SOURCE_COLMAP}/sparse/0/${file}" ]]; then
    echo "Missing COLMAP model file: ${SOURCE_COLMAP}/sparse/0/${file}" >&2
    exit 1
  fi
done
if [[ ! -f "${LIDAR_PLY}" ]]; then
  echo "Missing aligned LiDAR PLY: ${LIDAR_PLY}" >&2
  exit 1
fi
if [[ ! -f "${GS_ROOT}/scene/__init__.py" ]]; then
  echo "Missing gaussian-splatting submodule: ${GS_ROOT}" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"
DATASET_DIR="${DATASET_DIR:-${WORK_DIR}/dataset}"
MODEL_DIR="${MODEL_DIR:-${WORK_DIR}/output}"

MVA_MODE="${MVA_MODE:-robust_median}"
MAX_POINTS="${MAX_POINTS:-250000}"
MAX_COLOR_VIEWS="${MAX_COLOR_VIEWS:-0}"
VOXEL_SIZE="${VOXEL_SIZE:-0.10}"
PATCH_RADIUS="${PATCH_RADIUS:-1}"

if [[ ! -f "${DATASET_DIR}/mva_ligs_prepare_report.json" ]]; then
  "${PYTHON}" "${REPO_ROOT}/prepare_mva_ligs_dataset.py" \
    --source_colmap "${SOURCE_COLMAP}" \
    --lidar_ply "${LIDAR_PLY}" \
    --out_dataset "${DATASET_DIR}" \
    --mode "${MVA_MODE}" \
    --voxel_size "${VOXEL_SIZE}" \
    --max_points "${MAX_POINTS}" \
    --max_color_views "${MAX_COLOR_VIEWS}" \
    --patch_radius "${PATCH_RADIUS}"
else
  echo "[MVA-LiGS] Reusing prepared dataset: ${DATASET_DIR}" >&2
fi

MVA_ATTRS="${DATASET_DIR}/mva_ligs_attributes.npz"
if [[ ! -f "${MVA_ATTRS}" ]]; then
  echo "Missing MVA attributes: ${MVA_ATTRS}" >&2
  exit 1
fi

ITERATIONS="${ITERATIONS:-30000}"
RESOLUTION="${RESOLUTION:-1}"
DATA_DEVICE="${DATA_DEVICE:-cuda}"
LAMBDA_MVA_COLOR="${LAMBDA_MVA_COLOR:-0.01}"
MVA_COLOR_MIN_VIEWS="${MVA_COLOR_MIN_VIEWS:-4}"
MVA_COLOR_MAX_RGB_STD="${MVA_COLOR_MAX_RGB_STD:-80.0}"
MVA_COLOR_CONFIDENCE_POWER="${MVA_COLOR_CONFIDENCE_POWER:-1.0}"
MVA_COLOR_DECAY_START_ITER="${MVA_COLOR_DECAY_START_ITER:-0}"
MVA_COLOR_DECAY_UNTIL_ITER="${MVA_COLOR_DECAY_UNTIL_ITER:-1000}"
MVA_ANCHOR_MAX_POINTS="${MVA_ANCHOR_MAX_POINTS:-20000}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" "${REPO_ROOT}/train_mva_ligs.py" \
  -s "${DATASET_DIR}" \
  -m "${MODEL_DIR}" \
  --eval \
  --iterations "${ITERATIONS}" \
  --test_iterations "${ITERATIONS}" \
  --resolution "${RESOLUTION}" \
  --data_device "${DATA_DEVICE}" \
  --mva_color_prior_path "${MVA_ATTRS}" \
  --lambda_mva_color "${LAMBDA_MVA_COLOR}" \
  --mva_color_min_views "${MVA_COLOR_MIN_VIEWS}" \
  --mva_color_max_rgb_std "${MVA_COLOR_MAX_RGB_STD}" \
  --mva_color_confidence_power "${MVA_COLOR_CONFIDENCE_POWER}" \
  --mva_color_decay_start_iter "${MVA_COLOR_DECAY_START_ITER}" \
  --mva_color_decay_until_iter "${MVA_COLOR_DECAY_UNTIL_ITER}" \
  --mva_anchor_max_points "${MVA_ANCHOR_MAX_POINTS}" \
  "$@"
