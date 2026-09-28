#!/usr/bin/env python3
"""Render and evaluate an MVA-LiGS/3DGS model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm


DEFAULT_GS_ROOT = Path(os.environ.get("GS_ROOT", Path(__file__).resolve().parents[1] / "third_party" / "gaussian-splatting")).resolve()
DEFAULT_PYTHON = Path(sys.executable)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(DEFAULT_GS_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_GS_ROOT))

from lpipsPyTorch import lpips  # type: ignore
from utils.image_utils import psnr  # type: ignore
from utils.loss_utils import ssim  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Render and evaluate an MVA-LiGS/3DGS model")
    p.add_argument("--model", type=Path, required=True, help="MVA-LiGS/3DGS model directory containing cfg_args.")
    p.add_argument("--iteration", type=int, default=-1, help="Iteration to render. Use -1 for latest.")
    p.add_argument("--gs_root", type=Path, default=DEFAULT_GS_ROOT)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--device", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    p.add_argument(
        "--metrics-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="PSNR/SSIM/LPIPS compute device. Use 'cpu' to reduce GPU memory usage.",
    )
    p.add_argument("--skip_render", action="store_true")
    p.add_argument("--skip_metrics", action="store_true")
    p.add_argument("--compare_baseline", type=Path, default=None, help="Optional baseline model dir with results.json.")
    p.add_argument("--summary_json", type=Path, default=None)
    return p.parse_args()


def run(cmd: List[str], cwd: Path, cuda_visible_devices: Optional[str]) -> None:
    env = dict(**__import__("os").environ)
    if cuda_visible_devices is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    print("[MVA-LiGS eval]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def metric_torch_device(metrics_mode: str) -> torch.device:
    if metrics_mode == "cpu":
        return torch.device("cpu")
    if metrics_mode == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def latest_method(results: Dict[str, Any]) -> Optional[str]:
    if not results:
        return None
    def key_fn(k: str) -> int:
        try:
            return int(k.split("_")[-1])
        except Exception:
            return -1
    return sorted(results.keys(), key=key_fn)[-1]


def metric_delta(current: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k in ("PSNR", "SSIM", "LPIPS"):
        if k in current and k in baseline:
            out[k] = float(current[k]) - float(baseline[k])
    return out


def evaluate_split_method(model: Path, split: str, method: str, device: torch.device):
    method_dir = model / split / method
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if not renders_dir.is_dir() or not gt_dir.is_dir():
        return None, None
    image_names = sorted(f for f in os.listdir(renders_dir) if not f.startswith("."))
    ssims: List[float] = []
    psnrs: List[float] = []
    lpipss: List[float] = []
    for fname in tqdm(image_names, desc=f"{split}/{method} metrics"):
        render = Image.open(renders_dir / fname).convert("RGB")
        gt_im = Image.open(gt_dir / fname).convert("RGB")
        r = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].to(device)
        g = tf.to_tensor(gt_im).unsqueeze(0)[:, :3, :, :].to(device)
        ssims.append(float(ssim(r, g).detach().cpu()))
        psnrs.append(float(psnr(r, g).detach().cpu()))
        lpipss.append(float(lpips(r, g, net_type="vgg").detach().cpu()))
        del r, g
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics = {
        "SSIM": float(sum(ssims) / max(len(ssims), 1)),
        "PSNR": float(sum(psnrs) / max(len(psnrs), 1)),
        "LPIPS": float(sum(lpipss) / max(len(lpipss), 1)),
    }
    per_view = {
        "SSIM": {name: v for name, v in zip(image_names, ssims)},
        "PSNR": {name: v for name, v in zip(image_names, psnrs)},
        "LPIPS": {name: v for name, v in zip(image_names, lpipss)},
    }
    return metrics, per_view


def evaluate_train_test(model: Path, metrics_mode: str) -> Dict[str, Any]:
    device = metric_torch_device(metrics_mode)
    out: Dict[str, Any] = {}
    per_view: Dict[str, Any] = {}
    for split in ("train", "test"):
        split_dir = model / split
        if not split_dir.is_dir():
            continue
        out[split] = {}
        per_view[split] = {}
        for method in sorted(os.listdir(split_dir)):
            metrics, pv = evaluate_split_method(model, split, method, device)
            if metrics is not None:
                out[split][method] = metrics
                per_view[split][method] = pv
    (model / "results_train_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (model / "per_view_train_test.json").write_text(json.dumps(per_view, indent=2), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    model = args.model.resolve()
    gs_root = args.gs_root.resolve()
    py = args.python.resolve()

    if not model.exists():
        raise FileNotFoundError(f"Model directory not found: {model}")
    render_script = REPO_ROOT / "render_mva_ligs.py"
    if not render_script.exists():
        raise FileNotFoundError(f"MVA-LiGS render script not found: {render_script}")
    if not py.exists():
        raise FileNotFoundError(f"Python executable not found: {py}")

    if not args.skip_render:
        render_cmd = [str(py), str(render_script), "-m", str(model)]
        if int(args.iteration) != -1:
            render_cmd += ["--iteration", str(int(args.iteration))]
        run(render_cmd, cwd=REPO_ROOT, cuda_visible_devices=str(args.device))

    split_results = {} if args.skip_metrics else evaluate_train_test(model, str(args.metrics_device))
    results = load_json(model / "results.json") or {}
    test_results = split_results.get("test", {})
    method = latest_method(test_results) or latest_method(results)
    current = test_results.get(method, results.get(method, {})) if method is not None else {}
    summary: Dict[str, Any] = {
        "model": str(model),
        "iteration": int(args.iteration),
        "method": method,
        "metrics_device": str(args.metrics_device),
        "metrics": current,
        "split_metrics": split_results,
        "results_json": str(model / "results.json"),
        "per_view_json": str(model / "per_view.json"),
        "results_train_test_json": str(model / "results_train_test.json"),
        "per_view_train_test_json": str(model / "per_view_train_test.json"),
    }

    if args.compare_baseline is not None and not args.skip_metrics:
        baseline_split_results = evaluate_train_test(args.compare_baseline, str(args.metrics_device))
        baseline_results = load_json(args.compare_baseline / "results.json") or {}
        baseline_test_results = baseline_split_results.get("test", {})
        baseline_method = latest_method(baseline_test_results) or latest_method(baseline_results)
        baseline = (
            baseline_test_results.get(baseline_method, baseline_results.get(baseline_method, {}))
            if baseline_method is not None
            else {}
        )
        summary["baseline"] = {
            "model": str(args.compare_baseline.resolve()),
            "method": baseline_method,
            "metrics": baseline,
            "split_metrics": baseline_split_results,
            "delta_current_minus_baseline": metric_delta(current, baseline),
        }

    summary_path = args.summary_json
    if summary_path is None:
        summary_path = model / "mva_ligs_eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[MVA-LiGS eval] Summary:", summary_path, flush=True)
    if current:
        print("[MVA-LiGS eval] Metrics:", json.dumps(current, indent=2), flush=True)
    if split_results:
        print("[MVA-LiGS eval] Train/test metrics:", json.dumps(split_results, indent=2), flush=True)
    if "baseline" in summary:
        print(
            "[MVA-LiGS eval] Delta current-baseline:",
            json.dumps(summary["baseline"]["delta_current_minus_baseline"], indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()
