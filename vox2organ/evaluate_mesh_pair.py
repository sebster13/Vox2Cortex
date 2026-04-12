#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import trimesh
from pytorch3d.structures import Meshes

from utils.eval_metrics import SurfaceDistance, SelfIntersections


def load_mesh_as_torch3d(mesh_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a single surface mesh using trimesh and return (verts, faces) as torch tensors.
    """
    mesh = trimesh.load(mesh_path, process=True)
    if isinstance(mesh, trimesh.Scene):
        # Merge geometry into one TriMesh if a scene
        geom = [g for g in mesh.dump().geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geom) == 0:
            raise ValueError(f"No triangular geometry found in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(geom)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type for: {mesh_path}")

    verts = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(device)
    faces = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)).to(device)
    return verts, faces


def build_meshes(
    mesh_paths: List[str],
    device: torch.device
) -> Meshes:
    """
    Build a PyTorch3D Meshes object from one or more mesh files.
    Each file becomes one sub-mesh entry (matching how your SurfaceDistance iterates).
    """
    verts_list, faces_list = [], []
    for p in mesh_paths:
        v, f = load_mesh_as_torch3d(p, device)
        verts_list.append(v)
        faces_list.append(f)
    return Meshes(verts_list, faces_list)


def run_evaluation(
    pred_paths: List[str],
    gt_paths: List[str],
    labels: List[str],
    n_points: int,
    device: torch.device,
    run_self_intersections: bool = False
) -> pd.DataFrame:
    """
    Run SurfaceDistance (and optionally SelfIntersections) on lists of meshes.

    pred_paths and gt_paths must be the same length as labels,
    each entry representing one corresponding structure.
    """
    assert len(pred_paths) == len(gt_paths) == len(labels), \
        "pred_paths, gt_paths, and labels must have the same length."

    # Build PyTorch3D Meshes
    mesh_pred = build_meshes(pred_paths, device)
    mesh_gt = build_meshes(gt_paths, device)

    # Surface distances
    sd = SurfaceDistance()
    sd.n_points = n_points  # override if desired

    # Your EvalMetric signature requires voxel args; we pass dummies.
    dummy_vox = torch.zeros(1, dtype=torch.int16, device=device)  # unused
    res_sd = sd(
        mesh_pred=mesh_pred,
        mesh_gt=mesh_gt,
        n_m_classes=len(labels),
        mesh_label_names=labels,
        voxel_pred=dummy_vox,
        voxel_gt=dummy_vox,
        n_v_classes=1,
        voxel_label_names=[],
    )

    frames = [pd.DataFrame(res_sd)]

    if run_self_intersections:
        si = SelfIntersections()
        res_si = si(
            mesh_pred=mesh_pred,
            mesh_gt=mesh_gt,
            n_m_classes=len(labels),
            mesh_label_names=labels,
            voxel_pred=dummy_vox,
            voxel_gt=dummy_vox,
            n_v_classes=1,
            voxel_label_names=[],
        )
        frames.append(pd.DataFrame(res_si))

    out_df = pd.concat(frames, ignore_index=True)
    return out_df


def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate two meshes (pred vs. gt) using the project's metrics."
    )
    ap.add_argument("--pred", nargs="+", required=True,
                    help="Path(s) to predicted/processed mesh(es). One per label.")
    ap.add_argument("--gt", nargs="+", required=True,
                    help="Path(s) to ground-truth/original mesh(es). One per label.")
    ap.add_argument("--labels", nargs="+", required=False, default=None,
                    help="Labels for each structure. Defaults to file basenames.")
    ap.add_argument("--n_points", type=int, default=10000,
                    help="Number of surface samples per mesh for distance metrics.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cpu", "cuda"], help="Device to run evaluation on.")
    ap.add_argument("--self_intersections", action="store_true",
                    help="Also compute SelfIntersections for predicted meshes.")
    ap.add_argument("--out_csv", type=str, default=None, help="Optional path to write CSV.")
    ap.add_argument("--out_json", type=str, default=None, help="Optional path to write JSON.")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.labels is None:
        if len(args.pred) != len(args.gt):
            raise ValueError("If labels are omitted, pred and gt counts must match.")
        labels = [os.path.basename(p) for p in args.pred]
    else:
        labels = args.labels
        if not (len(labels) == len(args.pred) == len(args.gt)):
            raise ValueError("labels, pred, and gt must have the same length.")

    df = run_evaluation(
        pred_paths=args.pred,
        gt_paths=args.gt,
        labels=labels,
        n_points=args.n_points,
        device=device,
        run_self_intersections=args.self_intersections
    )

    # Print
    pd.set_option("display.max_rows", None)
    print(df)

    # Save
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
    if args.out_json:
        # Keep a simple list of dicts for portability
        with open(args.out_json, "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)


if __name__ == "__main__":
    main()