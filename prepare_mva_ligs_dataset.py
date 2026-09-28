#!/usr/bin/env python3
"""Prepare a Multi-View Attribute corrected LiDAR-GS dataset.

It keeps LiDAR points as the Gaussian geometry anchors, but estimates their
initial RGB attributes from multi-view image evidence.  v1 uses robust median
aggregation.  v2 adds edge-aware, consistency-gated aggregation so unreliable
multi-view color does not blindly overwrite safer per-view evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from tqdm import tqdm


GAUSSIAN_SPLATTING_ROOT = Path(os.environ.get("GS_ROOT", Path(__file__).resolve().parent / "third_party" / "gaussian-splatting")).resolve()
if str(GAUSSIAN_SPLATTING_ROOT) not in sys.path:
    sys.path.insert(0, str(GAUSSIAN_SPLATTING_ROOT))

from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary  # type: ignore


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def read_lidar_points_and_colors(
    path: Path,
    voxel_size: float,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray | None]:
    pcd = o3d.io.read_point_cloud(str(path))
    if float(voxel_size) > 0.0:
        pcd = pcd.voxel_down_sample(float(voxel_size))

    pts = np.asarray(pcd.points, dtype=np.float32)
    colors = None
    if np.asarray(pcd.colors).shape[0] == pts.shape[0]:
        colors = (np.asarray(pcd.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)

    if pts.shape[0] == 0:
        raise RuntimeError(f"Empty LiDAR point cloud: {path}")

    if int(max_points) > 0 and pts.shape[0] > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]

    return pts.astype(np.float32), colors


def camera_params(camera) -> Tuple[float, float, float, float, np.ndarray]:
    p = np.asarray(camera.params, dtype=np.float64)
    model = str(camera.model)
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        return float(f), float(f), float(cx), float(cy), np.zeros(4, dtype=np.float64)
    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        return float(fx), float(fy), float(cx), float(cy), np.zeros(4, dtype=np.float64)
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p[:4]
        return float(f), float(f), float(cx), float(cy), np.asarray([k1, 0.0, 0.0, 0.0])
    if model == "RADIAL":
        f, cx, cy, k1, k2 = p[:5]
        return float(f), float(f), float(cx), float(cy), np.asarray([k1, k2, 0.0, 0.0])
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        return float(fx), float(fy), float(cx), float(cy), np.asarray([k1, k2, p1, p2])
    raise RuntimeError(f"Unsupported COLMAP camera model: {model}")


def project(points: np.ndarray, image, camera) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = qvec2rotmat(image.qvec).astype(np.float64)
    t = np.asarray(image.tvec, dtype=np.float64)
    pc = (r @ points.astype(np.float64).T).T + t[None, :]
    z = pc[:, 2]
    x = pc[:, 0] / np.maximum(z, 1e-8)
    y = pc[:, 1] / np.maximum(z, 1e-8)

    fx, fy, cx, cy, dist = camera_params(camera)
    if np.any(dist != 0.0):
        k1, k2, p1, p2 = dist
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x, y = (
            x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        )

    u = fx * x + cx
    v = fy * y + cy
    valid = (
        (z > 1e-5)
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (v >= 0.0)
        & (u < float(camera.width))
        & (v < float(camera.height))
    )
    return np.stack([u, v], axis=1), z.astype(np.float32), valid


def sample_patch_median(rgb: np.ndarray, u: np.ndarray, v: np.ndarray, radius: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if int(radius) <= 0:
        return rgb[v.clip(0, h - 1), u.clip(0, w - 1)]

    patch_values = []
    for dy in range(-int(radius), int(radius) + 1):
        vv = (v + dy).clip(0, h - 1)
        for dx in range(-int(radius), int(radius) + 1):
            uu = (u + dx).clip(0, w - 1)
            patch_values.append(rgb[vv, uu])
    patch = np.stack(patch_values, axis=0).astype(np.float32)
    return np.median(patch, axis=0).clip(0, 255).astype(np.uint8)


def sample_bilinear_rgb(rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    uf = np.asarray(u, dtype=np.float32).clip(0.0, float(w - 1))
    vf = np.asarray(v, dtype=np.float32).clip(0.0, float(h - 1))
    x0 = np.floor(uf).astype(np.int64)
    y0 = np.floor(vf).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (uf - x0.astype(np.float32))[:, None]
    wy = (vf - y0.astype(np.float32))[:, None]

    c00 = rgb[y0, x0].astype(np.float32)
    c10 = rgb[y0, x1].astype(np.float32)
    c01 = rgb[y1, x0].astype(np.float32)
    c11 = rgb[y1, x1].astype(np.float32)
    c0 = c00 * (1.0 - wx) + c10 * wx
    c1 = c01 * (1.0 - wx) + c11 * wx
    return (c0 * (1.0 - wy) + c1 * wy).clip(0, 255).astype(np.uint8)


def sample_bilinear_patch_median(rgb: np.ndarray, u: np.ndarray, v: np.ndarray, radius: int) -> np.ndarray:
    if int(radius) <= 0:
        return sample_bilinear_rgb(rgb, u, v)
    values = []
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            values.append(sample_bilinear_rgb(rgb, np.asarray(u) + dx, np.asarray(v) + dy))
    patch = np.stack(values, axis=0).astype(np.float32)
    return np.median(patch, axis=0).clip(0, 255).astype(np.uint8)


def build_lidar_zbuffer(
    points: np.ndarray,
    image,
    camera,
    height: int,
    width: int,
    chunk_size: int,
) -> np.ndarray:
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    chunk = int(max(50_000, chunk_size))
    for start in range(0, int(points.shape[0]), chunk):
        end = min(int(points.shape[0]), start + chunk)
        uv, z, valid = project(points[start:end], image, camera)
        if not np.any(valid):
            continue
        local = np.nonzero(valid)[0]
        px = np.round(uv[local, 0]).astype(np.int64).clip(0, width - 1)
        py = np.round(uv[local, 1]).astype(np.int64).clip(0, height - 1)
        np.minimum.at(zbuf, (py, px), z[local])
    return zbuf


def zbuffer_visibility(
    zbuf: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    radius: int,
    depth_margin: float,
) -> np.ndarray:
    h, w = zbuf.shape[:2]
    px = np.round(u).astype(np.int64).clip(0, w - 1)
    py = np.round(v).astype(np.int64).clip(0, h - 1)
    nearest = np.full(px.shape, np.inf, dtype=np.float32)
    for dy in range(-int(radius), int(radius) + 1):
        yy = (py + dy).clip(0, h - 1)
        for dx in range(-int(radius), int(radius) + 1):
            xx = (px + dx).clip(0, w - 1)
            nearest = np.minimum(nearest, zbuf[yy, xx])
    return np.isfinite(nearest) & (z <= nearest + float(depth_margin))


def compute_edge_map(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    denom = np.percentile(mag, 95.0)
    if not np.isfinite(denom) or denom <= 1e-6:
        return np.zeros_like(mag, dtype=np.float32)
    return np.clip(mag / float(denom), 0.0, 1.0).astype(np.float32)


def masked_nanmean(values: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    arr = values.astype(np.float32)
    mask = valid
    if arr.ndim == 3 and mask.ndim == 2:
        mask = mask[:, :, None]
    mask = np.broadcast_to(mask, arr.shape)
    arr[~mask] = np.nan
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def masked_nanmedian(values: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    arr = values.astype(np.float32)
    mask = valid
    if arr.ndim == 3 and mask.ndim == 2:
        mask = mask[:, :, None]
    mask = np.broadcast_to(mask, arr.shape)
    arr[~mask] = np.nan
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(arr, axis=axis)


def masked_rgb_std(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    mask = valid
    if arr.ndim == 3 and mask.ndim == 2:
        mask = mask[:, :, None]
    mask = np.broadcast_to(mask, arr.shape)
    arr[~mask] = np.nan
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanmean(np.nanstd(arr, axis=1), axis=1)
    std[~np.isfinite(std)] = 255.0
    return std.astype(np.float32)


def aggregate_multiview_colors(
    samples: np.ndarray,
    valid: np.ndarray,
    fallback_colors: np.ndarray,
    aggregate: str,
    min_blend_views: int,
    std_scale: float,
    blend_with_fallback: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_f = samples.astype(np.float32)
    sample_f[~valid] = np.nan
    counts = valid.sum(axis=1).astype(np.int32)

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if aggregate == "mean":
            mv_rgb = np.nanmean(sample_f, axis=1)
        else:
            mv_rgb = np.nanmedian(sample_f, axis=1)
        rgb_std = np.nanmean(np.nanstd(sample_f, axis=1), axis=1)

    has_any = counts > 0
    mv_rgb[~np.isfinite(mv_rgb)] = fallback_colors[~np.isfinite(mv_rgb)].astype(np.float32)
    rgb_std[~np.isfinite(rgb_std)] = 255.0

    view_conf = np.minimum(counts.astype(np.float32) / float(max(1, min_blend_views)), 1.0)
    color_conf = np.exp(-rgb_std.astype(np.float32) / float(max(1e-6, std_scale)))
    confidence = (view_conf * color_conf).clip(0.0, 1.0)

    final = fallback_colors.astype(np.float32)
    if blend_with_fallback:
        final[has_any] = (
            confidence[has_any, None] * mv_rgb[has_any]
            + (1.0 - confidence[has_any, None]) * fallback_colors[has_any].astype(np.float32)
        )
    else:
        final[has_any] = mv_rgb[has_any]
    final = final.clip(0, 255).astype(np.uint8)
    return final, counts, rgb_std.astype(np.float32), confidence.astype(np.float32)


def aggregate_edge_consistency_gated(
    center_samples: np.ndarray,
    patch_samples: np.ndarray,
    edge_samples: np.ndarray,
    valid: np.ndarray,
    fallback_colors: np.ndarray,
    min_stable_views: int,
    stable_std_quantile: float,
    edge_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    counts = valid.sum(axis=1).astype(np.int32)
    has_any = counts > 0

    center_mean = masked_nanmean(center_samples, valid, axis=1)
    center_median = masked_nanmedian(center_samples, valid, axis=1)
    patch_median = masked_nanmedian(patch_samples, valid, axis=1)
    rgb_std = masked_rgb_std(center_samples, valid)

    edge_f = edge_samples.astype(np.float32)
    edge_f[~valid] = np.nan
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        edge_mean = np.nanmean(edge_f, axis=1)
    edge_mean[~np.isfinite(edge_mean)] = 0.0

    center_mean[~np.isfinite(center_mean)] = fallback_colors[~np.isfinite(center_mean)].astype(np.float32)
    center_median[~np.isfinite(center_median)] = center_mean[~np.isfinite(center_median)]
    patch_median[~np.isfinite(patch_median)] = center_mean[~np.isfinite(patch_median)]

    if np.any(has_any):
        stable_std_thresh = float(np.quantile(rgb_std[has_any], float(stable_std_quantile)))
        edge_thresh = float(np.quantile(edge_mean[has_any], float(edge_quantile)))
    else:
        stable_std_thresh = 255.0
        edge_thresh = 1.0

    enough_views = counts >= int(min_stable_views)
    stable = has_any & enough_views & (rgb_std <= stable_std_thresh) & (edge_mean < edge_thresh)
    edge_sensitive = has_any & (edge_mean >= edge_thresh)
    unstable = has_any & ~(stable | edge_sensitive)

    final = fallback_colors.astype(np.float32)
    final[unstable] = center_mean[unstable]
    final[edge_sensitive] = center_median[edge_sensitive]
    final[stable] = patch_median[stable]

    view_conf = np.minimum(counts.astype(np.float32) / float(max(1, min_stable_views)), 1.0)
    std_conf = 1.0 - np.clip(rgb_std / max(stable_std_thresh, 1e-6), 0.0, 1.0)
    edge_conf = 1.0 - np.clip(edge_mean / max(edge_thresh, 1e-6), 0.0, 1.0)
    confidence = (view_conf * std_conf * np.maximum(edge_conf, 0.25)).clip(0.0, 1.0)

    group = np.zeros((counts.shape[0],), dtype=np.uint8)
    group[unstable] = 1
    group[edge_sensitive] = 2
    group[stable] = 3

    report = {
        "stable_std_thresh": stable_std_thresh,
        "edge_thresh": edge_thresh,
        "stable_count": int(np.count_nonzero(stable)),
        "edge_sensitive_count": int(np.count_nonzero(edge_sensitive)),
        "unstable_count": int(np.count_nonzero(unstable)),
        "uncolored_count": int(np.count_nonzero(~has_any)),
        "stable_ratio": float(np.mean(stable)),
        "edge_sensitive_ratio": float(np.mean(edge_sensitive)),
        "unstable_ratio": float(np.mean(unstable)),
    }
    return final.clip(0, 255).astype(np.uint8), counts, rgb_std, confidence.astype(np.float32), group, report


def aggregate_hf_aware(
    center_samples: np.ndarray,
    patch_samples: np.ndarray,
    edge_samples: np.ndarray,
    valid: np.ndarray,
    fallback_colors: np.ndarray,
    sample_hf_thresh: float,
    point_hf_quantile: float,
    consistency_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Aggregate RGB while preserving high-frequency projection samples.

    Low-HF samples keep v1's 3x3 patch-median stabilization.  HF samples use
    center pixels to avoid mixing foreground/background colors across edges.
    A point-level consistency gate prevents all views of unstable edge points
    from being treated as reliable detail.
    """

    counts = valid.sum(axis=1).astype(np.int32)
    has_any = counts > 0

    edge_f = edge_samples.astype(np.float32)
    edge_f[~valid] = np.nan
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        point_hf = np.nanmedian(edge_f, axis=1)
        point_hf_max = np.nanmax(edge_f, axis=1)
    point_hf[~np.isfinite(point_hf)] = 0.0
    point_hf_max[~np.isfinite(point_hf_max)] = 0.0

    rgb_std = masked_rgb_std(center_samples, valid)
    patch_median = masked_nanmedian(patch_samples, valid, axis=1)
    center_median = masked_nanmedian(center_samples, valid, axis=1)

    patch_median[~np.isfinite(patch_median)] = fallback_colors[~np.isfinite(patch_median)].astype(np.float32)
    center_median[~np.isfinite(center_median)] = patch_median[~np.isfinite(center_median)]

    if np.any(has_any):
        point_hf_thresh = float(np.quantile(point_hf[has_any], float(point_hf_quantile)))
        consistency_thresh = float(np.quantile(rgb_std[has_any], float(consistency_quantile)))
    else:
        point_hf_thresh = 1.0
        consistency_thresh = 255.0

    # Per-sample veto: on projected edge samples, use center RGB instead of
    # 3x3 patch median.  This keeps v1 behavior elsewhere.
    mixed_samples = patch_samples.copy()
    use_center_sample = valid & (edge_samples >= float(sample_hf_thresh))
    mixed_samples[use_center_sample] = center_samples[use_center_sample]
    mixed_median = masked_nanmedian(mixed_samples, valid, axis=1)
    mixed_median[~np.isfinite(mixed_median)] = patch_median[~np.isfinite(mixed_median)]

    high_hf = has_any & (point_hf >= point_hf_thresh)
    consistent = rgb_std <= consistency_thresh
    hf_consistent = high_hf & consistent
    hf_unstable = high_hf & ~consistent
    low_hf = has_any & ~high_hf

    final = fallback_colors.astype(np.float32)
    final[low_hf] = patch_median[low_hf]
    final[hf_consistent] = mixed_median[hf_consistent]
    # For unstable HF points, keep a conservative blend: enough center evidence
    # to reduce edge smearing, but not enough to fully trust noisy projections.
    final[hf_unstable] = 0.75 * patch_median[hf_unstable] + 0.25 * mixed_median[hf_unstable]

    group = np.zeros((counts.shape[0],), dtype=np.uint8)
    group[low_hf] = 1
    group[hf_unstable] = 2
    group[hf_consistent] = 3

    view_conf = np.minimum(counts.astype(np.float32) / 4.0, 1.0)
    std_conf = 1.0 - np.clip(rgb_std / max(consistency_thresh, 1e-6), 0.0, 1.0)
    hf_conf = np.clip(point_hf / max(point_hf_thresh, 1e-6), 0.0, 1.0)
    confidence = (0.5 * view_conf + 0.25 * std_conf + 0.25 * hf_conf).clip(0.0, 1.0)

    report = {
        "sample_hf_thresh": float(sample_hf_thresh),
        "point_hf_thresh": point_hf_thresh,
        "consistency_thresh": consistency_thresh,
        "low_hf_count": int(np.count_nonzero(low_hf)),
        "hf_consistent_count": int(np.count_nonzero(hf_consistent)),
        "hf_unstable_count": int(np.count_nonzero(hf_unstable)),
        "uncolored_count": int(np.count_nonzero(~has_any)),
        "center_sample_veto_count": int(np.count_nonzero(use_center_sample)),
        "valid_sample_count": int(np.count_nonzero(valid)),
        "center_sample_veto_ratio": float(np.count_nonzero(use_center_sample) / max(1, np.count_nonzero(valid))),
        "mean_point_hf_colored": float(np.mean(point_hf[has_any])) if np.any(has_any) else 0.0,
        "mean_point_hf_max_colored": float(np.mean(point_hf_max[has_any])) if np.any(has_any) else 0.0,
    }
    return final.clip(0, 255).astype(np.uint8), counts, rgb_std, confidence.astype(np.float32), group, report


def aggregate_trimmed_median(
    patch_samples: np.ndarray,
    valid: np.ndarray,
    fallback_colors: np.ndarray,
    trim_keep_ratio: float,
    min_trim_views: int,
    zbuffer_depth_margin: float,
    zbuffer_radius: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Remove view-wise RGB outliers before multiview median aggregation."""

    counts = valid.sum(axis=1).astype(np.int32)
    has_any = counts > 0
    rough = masked_nanmedian(patch_samples, valid, axis=1)
    rough[~np.isfinite(rough)] = fallback_colors[~np.isfinite(rough)].astype(np.float32)

    diff = patch_samples.astype(np.float32) - rough[:, None, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist[~valid] = np.nan

    keep = valid.copy()
    n = int(patch_samples.shape[0])
    min_views = int(max(1, min_trim_views))
    ratio = float(np.clip(trim_keep_ratio, 0.1, 1.0))

    for i in range(n):
        idx = np.nonzero(valid[i])[0]
        if idx.size <= min_views:
            continue
        k = int(np.ceil(float(idx.size) * ratio))
        k = max(min_views, min(k, int(idx.size)))
        order = idx[np.argsort(dist[i, idx])]
        keep[i, idx] = False
        keep[i, order[:k]] = True

    final = masked_nanmedian(patch_samples, keep, axis=1)
    final[~np.isfinite(final)] = rough[~np.isfinite(final)]
    rgb_std = masked_rgb_std(patch_samples, keep)

    kept_counts = keep.sum(axis=1).astype(np.int32)
    reject_counts = counts - kept_counts
    reject_ratio = reject_counts.astype(np.float32) / np.maximum(counts.astype(np.float32), 1.0)
    confidence = (
        np.minimum(kept_counts.astype(np.float32) / 4.0, 1.0)
        * (1.0 - np.clip(reject_ratio, 0.0, 1.0))
    ).clip(0.0, 1.0)

    group = np.zeros((n,), dtype=np.uint8)
    group[has_any & (reject_counts == 0)] = 1
    group[has_any & (reject_counts > 0)] = 2

    report = {
        "trim_keep_ratio": ratio,
        "min_trim_views": min_views,
        "trimmed_point_count": int(np.count_nonzero(reject_counts > 0)),
        "trimmed_point_ratio": float(np.mean(reject_counts > 0)),
        "total_valid_samples": int(np.count_nonzero(valid)),
        "total_kept_samples": int(np.count_nonzero(keep)),
        "total_rejected_samples": int(np.count_nonzero(valid) - np.count_nonzero(keep)),
        "sample_reject_ratio": float((np.count_nonzero(valid) - np.count_nonzero(keep)) / max(1, np.count_nonzero(valid))),
        "mean_kept_views_colored": float(np.mean(kept_counts[has_any])) if np.any(has_any) else 0.0,
        "mean_rejected_views_colored": float(np.mean(reject_counts[has_any])) if np.any(has_any) else 0.0,
    }
    return final.clip(0, 255).astype(np.uint8), kept_counts, rgb_std, confidence.astype(np.float32), group, report


def aggregate_view_cluster(
    patch_samples: np.ndarray,
    valid: np.ndarray,
    fallback_colors: np.ndarray,
    cluster_radius: float,
    min_cluster_views: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Aggregate only the dominant compact RGB cluster for each point."""

    counts = valid.sum(axis=1).astype(np.int32)
    has_any = counts > 0
    n = int(patch_samples.shape[0])
    final = fallback_colors.astype(np.float32).copy()
    kept_counts = np.zeros((n,), dtype=np.int32)
    rgb_std = np.full((n,), 255.0, dtype=np.float32)
    group = np.zeros((n,), dtype=np.uint8)

    radius = float(cluster_radius)
    min_views = int(max(1, min_cluster_views))
    clustered_points = 0
    fallback_to_all_points = 0
    total_valid_samples = int(np.count_nonzero(valid))
    total_kept_samples = 0

    for i in tqdm(range(n), desc="MVA view clusters"):
        idx = np.nonzero(valid[i])[0]
        if idx.size == 0:
            continue

        samples = patch_samples[i, idx].astype(np.float32)
        if idx.size == 1:
            chosen = np.ones((1,), dtype=bool)
        else:
            diff = samples[:, None, :] - samples[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            neighbor = dist <= radius
            neighbor_counts = neighbor.sum(axis=1)
            mean_dist = (dist * neighbor).sum(axis=1) / np.maximum(neighbor_counts, 1)
            best = np.lexsort((mean_dist, -neighbor_counts))[0]
            chosen = neighbor[best]
            if int(chosen.sum()) < min_views:
                chosen = np.ones((idx.size,), dtype=bool)
                fallback_to_all_points += 1
            else:
                clustered_points += 1

        chosen_samples = samples[chosen]
        final[i] = np.median(chosen_samples, axis=0)
        kept_counts[i] = int(chosen_samples.shape[0])
        rgb_std[i] = float(np.mean(np.std(chosen_samples, axis=0)))
        total_kept_samples += int(chosen_samples.shape[0])
        group[i] = 2 if chosen_samples.shape[0] < idx.size else 1

    confidence = (
        np.minimum(kept_counts.astype(np.float32) / float(max(1, min_views)), 1.0)
        * np.exp(-rgb_std.astype(np.float32) / 40.0)
    ).clip(0.0, 1.0)
    confidence[~has_any] = 0.0

    report = {
        "cluster_radius": radius,
        "min_cluster_views": min_views,
        "clustered_point_count": int(clustered_points),
        "clustered_point_ratio": float(clustered_points / max(1, n)),
        "fallback_to_all_point_count": int(fallback_to_all_points),
        "fallback_to_all_point_ratio": float(fallback_to_all_points / max(1, n)),
        "total_valid_samples": int(total_valid_samples),
        "total_kept_samples": int(total_kept_samples),
        "sample_keep_ratio": float(total_kept_samples / max(1, total_valid_samples)),
        "mean_kept_views_colored": float(np.mean(kept_counts[has_any])) if np.any(has_any) else 0.0,
    }
    return final.clip(0, 255).astype(np.uint8), kept_counts, rgb_std, confidence.astype(np.float32), group, report


def colorize_points_mva(
    points: np.ndarray,
    fallback_colors: np.ndarray,
    sparse_dir: Path,
    image_dir: Path,
    max_views: int,
    chunk_size: int,
    patch_radius: int,
    aggregate: str,
    min_blend_views: int,
    std_scale: float,
    blend_with_fallback: bool,
    mode: str,
    min_stable_views: int,
    stable_std_quantile: float,
    edge_quantile: float,
    sample_hf_thresh: float,
    point_hf_quantile: float,
    consistency_quantile: float,
    trim_keep_ratio: float,
    min_trim_views: int,
    cluster_radius: float,
    min_cluster_views: int,
    zbuffer_depth_margin: float,
    zbuffer_radius: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    cameras = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    images = sorted(read_extrinsics_binary(str(sparse_dir / "images.bin")).values(), key=lambda x: x.name)
    if int(max_views) > 0 and len(images) > int(max_views):
        ids = np.linspace(0, len(images) - 1, int(max_views)).round().astype(np.int64).tolist()
        images = [images[i] for i in ids]

    n = int(points.shape[0])
    v_count = len(images)
    center_samples = np.zeros((n, v_count, 3), dtype=np.uint8)
    patch_samples = np.zeros((n, v_count, 3), dtype=np.uint8)
    edge_samples = np.zeros((n, v_count), dtype=np.float32)
    valid_samples = np.zeros((n, v_count), dtype=bool)
    chunk = int(max(50_000, chunk_size))
    zbuffer_report = {
        "zbuffer_input_valid_samples": 0,
        "zbuffer_visible_samples": 0,
        "zbuffer_rejected_samples": 0,
        "zbuffer_depth_margin": float(zbuffer_depth_margin),
        "zbuffer_radius": int(zbuffer_radius),
    }

    for view_idx, img in enumerate(tqdm(images, desc="MVA color views")):
        camera = cameras[img.camera_id]
        bgr = cv2.imread(str(image_dir / img.name), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        edge_map = compute_edge_map(rgb) if mode in {"edge_consistency_gate", "hf_aware"} else None
        h, w = rgb.shape[:2]
        zbuf = None
        if mode == "zbuffer_bilinear":
            zbuf = build_lidar_zbuffer(points, img, camera, h, w, chunk)

        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            uv, z, valid = project(points[start:end], img, camera)
            if not np.any(valid):
                continue
            local = np.nonzero(valid)[0]
            uf = uv[local, 0]
            vf = uv[local, 1]
            if zbuf is not None:
                zbuffer_report["zbuffer_input_valid_samples"] += int(local.size)
                visible = zbuffer_visibility(
                    zbuf,
                    uf,
                    vf,
                    z[local],
                    radius=int(zbuffer_radius),
                    depth_margin=float(zbuffer_depth_margin),
                )
                if not np.any(visible):
                    zbuffer_report["zbuffer_rejected_samples"] += int(local.size)
                    continue
                zbuffer_report["zbuffer_visible_samples"] += int(np.count_nonzero(visible))
                zbuffer_report["zbuffer_rejected_samples"] += int(local.size - np.count_nonzero(visible))
                local = local[visible]
                uf = uf[visible]
                vf = vf[visible]

            u = np.round(uf).astype(np.int64).clip(0, w - 1)
            v = np.round(vf).astype(np.int64).clip(0, h - 1)
            global_idx = local + start
            if mode == "zbuffer_bilinear":
                center_samples[global_idx, view_idx] = sample_bilinear_rgb(rgb, uf, vf)
                patch_samples[global_idx, view_idx] = sample_bilinear_patch_median(rgb, uf, vf, int(patch_radius))
            else:
                center_samples[global_idx, view_idx] = rgb[v, u]
                patch_samples[global_idx, view_idx] = sample_patch_median(rgb, u, v, int(patch_radius))
            if edge_map is not None:
                edge_samples[global_idx, view_idx] = edge_map[v, u]
            valid_samples[global_idx, view_idx] = True

    if mode == "edge_consistency_gate":
        return aggregate_edge_consistency_gated(
            center_samples=center_samples,
            patch_samples=patch_samples,
            edge_samples=edge_samples,
            valid=valid_samples,
            fallback_colors=fallback_colors,
            min_stable_views=int(min_stable_views),
            stable_std_quantile=float(stable_std_quantile),
            edge_quantile=float(edge_quantile),
        )

    if mode == "hf_aware":
        return aggregate_hf_aware(
            center_samples=center_samples,
            patch_samples=patch_samples,
            edge_samples=edge_samples,
            valid=valid_samples,
            fallback_colors=fallback_colors,
            sample_hf_thresh=float(sample_hf_thresh),
            point_hf_quantile=float(point_hf_quantile),
            consistency_quantile=float(consistency_quantile),
        )

    if mode == "trimmed_median":
        return aggregate_trimmed_median(
            patch_samples=patch_samples,
            valid=valid_samples,
            fallback_colors=fallback_colors,
            trim_keep_ratio=float(trim_keep_ratio),
            min_trim_views=int(min_trim_views),
        )

    if mode == "view_cluster":
        return aggregate_view_cluster(
            patch_samples=patch_samples,
            valid=valid_samples,
            fallback_colors=fallback_colors,
            cluster_radius=float(cluster_radius),
            min_cluster_views=int(min_cluster_views),
        )

    if mode == "zbuffer_bilinear":
        colors, counts, rgb_std, confidence = aggregate_multiview_colors(
            samples=patch_samples,
            valid=valid_samples,
            fallback_colors=fallback_colors,
            aggregate=aggregate,
            min_blend_views=int(min_blend_views),
            std_scale=float(std_scale),
            blend_with_fallback=bool(blend_with_fallback),
        )
        zbuffer_report["zbuffer_reject_ratio"] = float(
            zbuffer_report["zbuffer_rejected_samples"] / max(1, zbuffer_report["zbuffer_input_valid_samples"])
        )
        return colors, counts, rgb_std, confidence, np.zeros((n,), dtype=np.uint8), zbuffer_report

    colors, counts, rgb_std, confidence = aggregate_multiview_colors(
        samples=patch_samples,
        valid=valid_samples,
        fallback_colors=fallback_colors,
        aggregate=aggregate,
        min_blend_views=int(min_blend_views),
        std_scale=float(std_scale),
        blend_with_fallback=bool(blend_with_fallback),
    )
    return colors, counts, rgb_std, confidence, np.zeros((n,), dtype=np.uint8), {}


def write_points3d_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(points, dtype=np.float32)
    arr = np.empty(points.shape[0], dtype=dtype)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["nx"], arr["ny"], arr["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MVA-LiGS dataset")
    parser.add_argument("--source_colmap", required=True)
    parser.add_argument("--lidar_ply", required=True)
    parser.add_argument("--out_dataset", required=True)
    parser.add_argument("--voxel_size", type=float, default=0.10)
    parser.add_argument("--max_points", type=int, default=250000)
    parser.add_argument("--max_color_views", type=int, default=38)
    parser.add_argument("--color_chunk_size", type=int, default=120000)
    parser.add_argument("--patch_radius", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=["robust_median", "edge_consistency_gate", "hf_aware", "trimmed_median", "zbuffer_bilinear", "view_cluster"],
        default="robust_median",
    )
    parser.add_argument("--aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--min_blend_views", type=int, default=4)
    parser.add_argument("--std_scale", type=float, default=40.0)
    parser.add_argument("--min_stable_views", type=int, default=4)
    parser.add_argument("--stable_std_quantile", type=float, default=0.45)
    parser.add_argument("--edge_quantile", type=float, default=0.75)
    parser.add_argument("--sample_hf_thresh", type=float, default=0.35)
    parser.add_argument("--point_hf_quantile", type=float, default=0.70)
    parser.add_argument("--consistency_quantile", type=float, default=0.70)
    parser.add_argument("--trim_keep_ratio", type=float, default=0.75)
    parser.add_argument("--min_trim_views", type=int, default=3)
    parser.add_argument("--cluster_radius", type=float, default=35.0)
    parser.add_argument("--min_cluster_views", type=int, default=3)
    parser.add_argument("--zbuffer_depth_margin", type=float, default=0.20)
    parser.add_argument("--zbuffer_radius", type=int, default=1)
    parser.add_argument("--blend_with_lidar_fallback", action="store_true", default=False)
    parser.add_argument("--copy_sparse_bins", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    source = Path(args.source_colmap).resolve()
    lidar_path = Path(args.lidar_ply).resolve()
    out = Path(args.out_dataset).resolve()
    sparse_src = source / "sparse" / "0"
    sparse_dst = out / "sparse" / "0"
    image_src = source / "images"
    image_dst = out / "images"

    out.mkdir(parents=True, exist_ok=True)
    link_or_copy(image_src, image_dst, copy=False)
    link_or_copy(sparse_src / "cameras.bin", sparse_dst / "cameras.bin", copy=bool(args.copy_sparse_bins))
    link_or_copy(sparse_src / "images.bin", sparse_dst / "images.bin", copy=bool(args.copy_sparse_bins))

    points, lidar_colors = read_lidar_points_and_colors(
        lidar_path,
        float(args.voxel_size),
        int(args.max_points),
        int(args.seed),
    )
    fallback = lidar_colors if lidar_colors is not None else np.full((points.shape[0], 3), 128, dtype=np.uint8)
    colors, view_counts, rgb_std, confidence, group, group_report = colorize_points_mva(
        points=points,
        fallback_colors=fallback,
        sparse_dir=sparse_src,
        image_dir=image_src,
        max_views=int(args.max_color_views),
        chunk_size=int(args.color_chunk_size),
        patch_radius=int(args.patch_radius),
        aggregate=str(args.aggregate),
        min_blend_views=int(args.min_blend_views),
        std_scale=float(args.std_scale),
        blend_with_fallback=bool(args.blend_with_lidar_fallback and lidar_colors is not None),
        mode=str(args.mode),
        min_stable_views=int(args.min_stable_views),
        stable_std_quantile=float(args.stable_std_quantile),
        edge_quantile=float(args.edge_quantile),
        sample_hf_thresh=float(args.sample_hf_thresh),
        point_hf_quantile=float(args.point_hf_quantile),
        consistency_quantile=float(args.consistency_quantile),
        trim_keep_ratio=float(args.trim_keep_ratio),
        min_trim_views=int(args.min_trim_views),
        cluster_radius=float(args.cluster_radius),
        min_cluster_views=int(args.min_cluster_views),
        zbuffer_depth_margin=float(args.zbuffer_depth_margin),
        zbuffer_radius=int(args.zbuffer_radius),
    )

    write_points3d_ply(sparse_dst / "points3D.ply", points, colors)
    np.savez_compressed(
        out / "mva_ligs_attributes.npz",
        view_counts=view_counts,
        rgb_std=rgb_std,
        confidence=confidence,
        colors=colors,
        group=group,
    )

    colored = view_counts > 0
    report = {
        "method": (
            "MVA-LiGS v3"
            if args.mode == "view_cluster"
            else "MVA-LiGS v3-old-hf-aware"
            if args.mode == "hf_aware"
            else "MVA-LiGS v4"
            if args.mode == "trimmed_median"
            else "MVA-LiGS v5b"
            if args.mode == "zbuffer_bilinear"
            else "MVA-LiGS v2"
            if args.mode == "edge_consistency_gate"
            else "MVA-LiGS v1"
        ),
        "source_colmap": str(source),
        "lidar_ply": str(lidar_path),
        "out_dataset": str(out),
        "voxel_size": float(args.voxel_size),
        "max_points": int(args.max_points),
        "points_written": int(points.shape[0]),
        "max_color_views": int(args.max_color_views),
        "patch_radius": int(args.patch_radius),
        "mode": str(args.mode),
        "aggregate": str(args.aggregate),
        "min_blend_views": int(args.min_blend_views),
        "std_scale": float(args.std_scale),
        "min_stable_views": int(args.min_stable_views),
        "stable_std_quantile": float(args.stable_std_quantile),
        "edge_quantile": float(args.edge_quantile),
        "sample_hf_thresh": float(args.sample_hf_thresh),
        "point_hf_quantile": float(args.point_hf_quantile),
        "consistency_quantile": float(args.consistency_quantile),
        "trim_keep_ratio": float(args.trim_keep_ratio),
        "min_trim_views": int(args.min_trim_views),
        "cluster_radius": float(args.cluster_radius),
        "min_cluster_views": int(args.min_cluster_views),
        "zbuffer_depth_margin": float(args.zbuffer_depth_margin),
        "zbuffer_radius": int(args.zbuffer_radius),
        "used_lidar_color_fallback": bool(lidar_colors is not None),
        "blend_with_lidar_fallback": bool(args.blend_with_lidar_fallback and lidar_colors is not None),
        "colored_points": int(np.count_nonzero(colored)),
        "colored_ratio": float(np.mean(colored)),
        "mean_color_views": float(np.mean(view_counts[colored])) if np.any(colored) else 0.0,
        "max_observed_views": int(np.max(view_counts)) if view_counts.size else 0,
        "mean_rgb_std_colored": float(np.mean(rgb_std[colored])) if np.any(colored) else 0.0,
        "mean_confidence_colored": float(np.mean(confidence[colored])) if np.any(colored) else 0.0,
        "group_report": group_report,
    }
    (out / "mva_ligs_prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
