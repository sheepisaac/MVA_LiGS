#!/usr/bin/env python3
"""Filter sparse outliers from an aligned LiDAR point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter aligned LiDAR PCD outliers")
    parser.add_argument("--input_ply", required=True)
    parser.add_argument("--output_ply", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--pre_voxel", type=float, default=0.05)
    parser.add_argument("--stat_nb_neighbors", type=int, default=30)
    parser.add_argument("--stat_std_ratio", type=float, default=2.0)
    parser.add_argument("--radius_nb_points", type=int, default=6)
    parser.add_argument("--radius", type=float, default=0.25)
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(str(args.input_ply))
    raw_count = len(pcd.points)
    if raw_count == 0:
        raise RuntimeError(f"Empty point cloud: {args.input_ply}")

    if float(args.pre_voxel) > 0.0:
        pcd = pcd.voxel_down_sample(float(args.pre_voxel))
    voxel_count = len(pcd.points)

    pcd_stat, stat_idx = pcd.remove_statistical_outlier(
        nb_neighbors=int(args.stat_nb_neighbors),
        std_ratio=float(args.stat_std_ratio),
    )
    stat_count = len(pcd_stat.points)

    pcd_radius, radius_idx = pcd_stat.remove_radius_outlier(
        nb_points=int(args.radius_nb_points),
        radius=float(args.radius),
    )
    radius_count = len(pcd_radius.points)

    out = Path(args.output_ply)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out), pcd_radius, write_ascii=False, compressed=False)

    report = {
        "input_ply": str(args.input_ply),
        "output_ply": str(args.output_ply),
        "raw_count": int(raw_count),
        "pre_voxel": float(args.pre_voxel),
        "after_pre_voxel_count": int(voxel_count),
        "stat_nb_neighbors": int(args.stat_nb_neighbors),
        "stat_std_ratio": float(args.stat_std_ratio),
        "after_statistical_count": int(stat_count),
        "radius_nb_points": int(args.radius_nb_points),
        "radius": float(args.radius),
        "after_radius_count": int(radius_count),
        "kept_ratio_vs_raw": float(radius_count / max(raw_count, 1)),
        "kept_ratio_vs_voxel": float(radius_count / max(voxel_count, 1)),
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
