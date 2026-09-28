#!/usr/bin/env python3
"""Align a LiDAR point cloud to a COLMAP SfM point cloud.

The input point clouds must already use compatible units and approximately the
same scale. The script estimates a rigid transform with FPFH + RANSAC followed
by point-to-plane ICP.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import open3d as o3d


def apply_transform_chunked(pcd: o3d.geometry.PointCloud, T: np.ndarray, chunk: int = 4_000_000) -> None:
    pts = np.asarray(pcd.points, dtype=np.float64)
    out = np.empty_like(pts, dtype=np.float64)
    T64 = np.asarray(T, dtype=np.float64)
    for i in range(0, pts.shape[0], chunk):
        sl = slice(i, min(i + chunk, pts.shape[0]))
        block = pts[sl]
        h = np.concatenate([block, np.ones((block.shape[0], 1), dtype=np.float64)], axis=1)
        out[sl] = (h @ T64.T)[:, :3]
    pcd.points = o3d.utility.Vector3dVector(out)


def fpfh_pipeline(
    pcd: o3d.geometry.PointCloud, voxel: float
) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
    )
    return down, fpfh


def pairwise_chamfer_stats(a: np.ndarray, b: np.ndarray, sample: int = 10000) -> Dict[str, float]:
    """One-way nearest: mean dist from a to b and b to a on random subsamples."""
    rng = np.random.default_rng(0)
    if a.shape[0] > sample:
        a = a[rng.choice(a.shape[0], size=sample, replace=False)]
    if b.shape[0] > sample:
        b = b[rng.choice(b.shape[0], size=sample, replace=False)]
    # Use numpy brute force too slow - KDTree
    t_a = o3d.geometry.PointCloud()
    t_a.points = o3d.utility.Vector3dVector(a.astype(np.float64))
    t_b = o3d.geometry.PointCloud()
    t_b.points = o3d.utility.Vector3dVector(b.astype(np.float64))
    kdt_a = o3d.geometry.KDTreeFlann(t_a)
    kdt_b = o3d.geometry.KDTreeFlann(t_b)
    d_ab = []
    for i in range(a.shape[0]):
        _, idx, d = kdt_b.search_knn_vector_3d(a[i], 1)
        d_ab.append(np.sqrt(d[0]))
    d_ba = []
    for i in range(b.shape[0]):
        _, idx, d = kdt_a.search_knn_vector_3d(b[i], 1)
        d_ba.append(np.sqrt(d[0]))
    d_ab = np.asarray(d_ab)
    d_ba = np.asarray(d_ba)
    return {
        "lidar_to_sfm_mean": float(np.mean(d_ab)),
        "lidar_to_sfm_median": float(np.median(d_ab)),
        "lidar_to_sfm_rmse": float(np.sqrt(np.mean(d_ab**2))),
        "sfm_to_lidar_mean": float(np.mean(d_ba)),
        "sfm_to_lidar_median": float(np.median(d_ba)),
        "sfm_to_lidar_rmse": float(np.sqrt(np.mean(d_ba**2))),
        "symmetric_chamfer_mean": float((np.mean(d_ab) + np.mean(d_ba)) / 2.0),
    }


def coverage_ratios(
    lidar_pts: np.ndarray,
    sfm_pts: np.ndarray,
    thresholds: Tuple[float, ...],
    max_pts: int = 50000,
) -> Dict[str, Any]:
    rng = np.random.default_rng(1)
    if lidar_pts.shape[0] > max_pts:
        lidar_pts = lidar_pts[rng.choice(lidar_pts.shape[0], size=max_pts, replace=False)]
    if sfm_pts.shape[0] > max_pts:
        sfm_pts = sfm_pts[rng.choice(sfm_pts.shape[0], size=max_pts, replace=False)]
    t_l = o3d.geometry.PointCloud()
    t_l.points = o3d.utility.Vector3dVector(lidar_pts.astype(np.float64))
    t_s = o3d.geometry.PointCloud()
    t_s.points = o3d.utility.Vector3dVector(sfm_pts.astype(np.float64))
    kdt_s = o3d.geometry.KDTreeFlann(t_s)
    kdt_l = o3d.geometry.KDTreeFlann(t_l)
    dists_l = []
    for i in range(lidar_pts.shape[0]):
        _, _, d2 = kdt_s.search_knn_vector_3d(lidar_pts[i], 1)
        dists_l.append(np.sqrt(d2[0]))
    dists_l = np.asarray(dists_l)
    dists_s = []
    for i in range(sfm_pts.shape[0]):
        _, _, d2 = kdt_l.search_knn_vector_3d(sfm_pts[i], 1)
        dists_s.append(np.sqrt(d2[0]))
    dists_s = np.asarray(dists_s)
    out: Dict[str, Any] = {
        "lidar_eval_points": int(lidar_pts.shape[0]),
        "sfm_eval_points": int(sfm_pts.shape[0]),
    }
    for th in thresholds:
        out[f"lidar_within_{th:.2f}m_ratio"] = float(np.mean(dists_l <= th))
        out[f"sfm_within_{th:.2f}m_ratio"] = float(np.mean(dists_s <= th))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Register LiDAR to COLMAP SfM coordinates")
    p.add_argument(
        "--lidar_input",
        type=Path,
        required=True,
        help="Input LiDAR point cloud in PLY format.",
    )
    p.add_argument(
        "--sfm_points",
        type=Path,
        required=True,
        help="COLMAP/SfM point cloud in PLY format.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory for the aligned PLY, transform, and JSON report.",
    )
    p.add_argument("--registration_voxel_size", type=float, default=0.1)
    p.add_argument("--eval_voxel_size", type=float, default=0.15)
    p.add_argument("--icp_threshold", type=float, default=0.2, help="ICP max correspondence distance (m)")
    p.add_argument(
        "--ransac_multiplier",
        type=float,
        default=1.5,
        help="RANSAC max correspondence = registration_voxel * this",
    )
    args = p.parse_args()

    lidar_in = args.lidar_input.resolve()
    sfm_path = args.sfm_points.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_path = out_dir / "lidar_aligned_to_sfm.ply"
    transform_path = out_dir / "lidar_to_sfm_transform.txt"
    report_path = out_dir / "registration_report.json"

    if not lidar_in.is_file():
        raise FileNotFoundError(f"LiDAR input not found: {lidar_in}")
    if not sfm_path.is_file():
        raise FileNotFoundError(f"COLMAP points3D.ply not found: {sfm_path}")

    t0 = time.time()
    print(f"[register_lidar_to_colmap] Loading LiDAR: {lidar_in}", flush=True)
    lidar_full = o3d.io.read_point_cloud(str(lidar_in))
    n_lidar_raw = len(lidar_full.points)
    if n_lidar_raw == 0:
        raise RuntimeError(f"Empty LiDAR point cloud: {lidar_in}")

    print(f"[register_lidar_to_colmap] Loading SfM: {sfm_path}", flush=True)
    sfm_full = o3d.io.read_point_cloud(str(sfm_path))
    n_sfm_raw = len(sfm_full.points)
    if n_sfm_raw == 0:
        raise RuntimeError(f"Empty SfM point cloud: {sfm_path}")

    rv = float(args.registration_voxel_size)
    ev = float(args.eval_voxel_size)
    lidar_reg = lidar_full.voxel_down_sample(rv)
    sfm_reg = sfm_full.voxel_down_sample(rv)
    n_lidar_reg = len(lidar_reg.points)
    n_sfm_reg = len(sfm_reg.points)
    print(
        f"[register_lidar_to_colmap] Downsample reg voxel={rv}: lidar {n_lidar_reg} pts, sfm {n_sfm_reg} pts",
        flush=True,
    )

    source, source_fpfh = fpfh_pipeline(lidar_reg, rv)
    target, target_fpfh = fpfh_pipeline(sfm_reg, rv)

    distance_threshold = rv * float(args.ransac_multiplier)
    T_global = np.eye(4, dtype=np.float64)
    global_fitness = 0.0
    global_rmse = 0.0
    try:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=estimation,
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
        T_global = np.asarray(ransac.transformation, dtype=np.float64)
        global_fitness = float(ransac.fitness)
        global_rmse = float(ransac.inlier_rmse)
        print(
            f"[register_lidar_to_colmap] RANSAC fitness={global_fitness:.6g} rmse={global_rmse:.6g}",
            flush=True,
        )
    except Exception as e:
        print(f"[register_lidar_to_colmap] RANSAC failed: {e}; using identity init.", flush=True)

    if global_fitness == 0.0 or not np.isfinite(global_fitness):
        print("[register_lidar_to_colmap] Using identity as ICP initial transform.", flush=True)
        T_global = np.eye(4, dtype=np.float64)

    # Point-to-plane ICP from global init
    source_icp = lidar_reg
    source_icp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rv * 2.0, max_nn=30))
    target_icp = sfm_reg
    target_icp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rv * 2.0, max_nn=30))

    icp_result = o3d.pipelines.registration.registration_icp(
        source_icp,
        target_icp,
        max_correspondence_distance=float(args.icp_threshold),
        init=T_global,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    T = np.asarray(icp_result.transformation, dtype=np.float64)
    icp_fitness = float(icp_result.fitness)
    icp_rmse = float(icp_result.inlier_rmse)
    print(f"[register_lidar_to_colmap] ICP fitness={icp_fitness:.6g} rmse={icp_rmse:.6g}", flush=True)

    # Metrics on eval voxel
    lidar_eval = lidar_full.voxel_down_sample(ev)
    sfm_eval = sfm_full.voxel_down_sample(ev)
    lidar_eval.transform(T)
    la = np.asarray(lidar_eval.points, dtype=np.float64)
    sa = np.asarray(sfm_eval.points, dtype=np.float64)
    dist_m = pairwise_chamfer_stats(la, sa, sample=12000)
    cov = coverage_ratios(la, sa, (0.10, 0.20, 0.50))

    q5 = lambda x: np.percentile(x, 5, axis=0)
    q95 = lambda x: np.percentile(x, 95, axis=0)
    la_box = la
    sa_box = sa
    robust_extent = {
        "lidar_diag": float(np.linalg.norm(q95(la_box) - q5(la_box))),
        "sfm_diag": float(np.linalg.norm(q95(sa_box) - q5(sa_box))),
        "lidar_q5": q5(la_box).tolist(),
        "lidar_q95": q95(la_box).tolist(),
        "sfm_q5": q5(sa_box).tolist(),
        "sfm_q95": q95(sa_box).tolist(),
    }

    # Write full aligned cloud (chunked transform)
    print(f"[register_lidar_to_colmap] Transforming full LiDAR ({n_lidar_raw} points)...", flush=True)
    lidar_out = o3d.geometry.PointCloud(lidar_full)
    apply_transform_chunked(lidar_out, T, chunk=4_000_000)
    o3d.io.write_point_cloud(str(aligned_path), lidar_out, write_ascii=False, compressed=False)
    print(f"[register_lidar_to_colmap] Wrote {aligned_path}", flush=True)

    np.savetxt(transform_path, T, fmt="%.10f")
    print(f"[register_lidar_to_colmap] Wrote {transform_path}", flush=True)

    inputs_block: Dict[str, Any] = {
        "lidar_input": str(lidar_in),
        "sfm_points": str(sfm_path),
    }

    report: Dict[str, Any] = {
        "timestamp_unix": time.time(),
        "inputs": inputs_block,
        "preprocess": {
            "registration_voxel_size": rv,
            "eval_voxel_size": ev,
            "lidar_points_raw": int(n_lidar_raw),
            "sfm_points_raw": int(n_sfm_raw),
            "lidar_points_reg": int(n_lidar_reg),
            "sfm_points_reg": int(n_sfm_reg),
        },
        "registration": {
            "global_method": "FPFH+RANSAC",
            "global_fitness": global_fitness,
            "global_inlier_rmse": global_rmse,
            "icp_method": "PointToPlane ICP",
            "icp_fitness": icp_fitness,
            "icp_inlier_rmse": icp_rmse,
            "transform_lidar_to_sfm_4x4": T.tolist(),
        },
        "distance_metrics_meters": dist_m,
        "coverage": cov,
        "robust_extent_5_95": robust_extent,
        "outputs": {
            "lidar_input_ply": str(lidar_in.resolve()),
            "lidar_aligned_ply": str(aligned_path.resolve()),
            "transform_txt": str(transform_path.resolve()),
            "report_json": str(report_path.resolve()),
        },
        "elapsed_sec": time.time() - t0,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[register_lidar_to_colmap] Done in {report['elapsed_sec']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
