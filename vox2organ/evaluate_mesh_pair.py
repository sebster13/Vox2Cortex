#!/usr/bin/env python3

"""
Evaluate cortical surface meshes.

Supported modes
---------------
1. gt-only
   Calculate general topological and geometric integrity metrics for one
   ground-truth surface and assign its configurable GT quality bin.

2. pair
   Calculate general metrics for both prediction and GT, prediction-versus-GT
   metrics, hard-failure screening, a configurable composite score, and a
   final reconstruction quality bin.

The output is one wide CSV row per hemisphere and surface.
"""

import argparse
import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import trimesh

try:
    import nibabel as nib
except ImportError:
    nib = None

from utils.eval_metrics import (
    GENERAL_METRIC_KEYS,
    SURFACE_DISTANCE_METRIC_KEYS,
    compute_binary_overlap,
    compute_correspondence_rmsd,
    compute_general_mesh_metrics,
    compute_surface_distance_metrics,
)


SCHEMA_VERSION = 2


@dataclass
class LoadedMesh:
    path: str
    vertices: np.ndarray
    faces: np.ndarray


def load_mesh_unprocessed(mesh_path: str) -> LoadedMesh:
    """
    Load a triangular mesh without topology-changing Trimesh processing.

    process=False is important because vertex merging, face removal, or
    topology repair would alter metrics such as Euler characteristic.
    """
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(
            "Mesh file does not exist: {}".format(mesh_path)
        )

    loaded = trimesh.load(
        mesh_path,
        process=False,
    )

    if isinstance(loaded, trimesh.Scene):
        geometry = list(loaded.geometry.values())

        if len(geometry) == 0:
            raise ValueError(
                "No geometry found in scene: {}".format(mesh_path)
            )

        try:
            loaded = loaded.dump(concatenate=True)
        except Exception:
            triangular_geometry = [
                item
                for item in geometry
                if isinstance(item, trimesh.Trimesh)
            ]

            if len(triangular_geometry) == 0:
                raise ValueError(
                    "No triangular geometry found in scene: {}"
                    .format(mesh_path)
                )

            loaded = trimesh.util.concatenate(
                triangular_geometry
            )

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(
            "Unsupported mesh type for: {}".format(mesh_path)
        )

    vertices = np.asarray(
        loaded.vertices,
        dtype=np.float64,
    )

    faces = np.asarray(
        loaded.faces,
        dtype=np.int64,
    )

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            "Vertices in {} do not have shape (V, 3)."
            .format(mesh_path)
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            "Faces in {} do not have shape (F, 3). "
            "The evaluator requires triangular meshes."
            .format(mesh_path)
        )

    return LoadedMesh(
        path=os.path.abspath(mesh_path),
        vertices=vertices,
        faces=faces,
    )


def deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge dictionaries.
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = copy.deepcopy(value)

    return result


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            "Quality configuration does not exist: {}"
            .format(config_path)
        )

    with open(config_path, "r") as file_handle:
        config = json.load(file_handle)

    return config


def resolve_config(
    raw_config: Dict[str, Any],
    hemisphere: str,
    surface: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Apply optional per-hemisphere/per-surface configuration overrides.

    Recognized override precedence is:

        *.*
        *.surface
        hemisphere.*
        hemisphere.surface

    Example:

        "overrides": {
            "L.pial": {
                "composite_score": {
                    "metrics": {
                        "ASSD": {
                            "mean": 0.4,
                            "std": 0.1,
                            "weight": 0.5
                        }
                    }
                }
            }
        }
    """
    config = copy.deepcopy(raw_config)
    overrides = config.pop("overrides", {})

    hemisphere = str(hemisphere)
    surface = str(surface)

    candidates = [
        "*.*",
        "*.{}".format(surface),
        "{}.*".format(hemisphere),
        "{}.{}".format(hemisphere, surface),
    ]

    applied = []

    for candidate in candidates:
        if candidate in overrides:
            config = deep_merge(
                config,
                overrides[candidate],
            )
            applied.append(candidate)

    profile_name = (
        ",".join(applied)
        if len(applied) > 0
        else "default"
    )

    return config, profile_name


def is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def prefix_metrics(
    metrics: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    return {
        "{}{}".format(prefix, key): value
        for key, value in metrics.items()
    }


def classify_lower_is_better_metric(
    metric_name: str,
    value: Any,
    rules: Any,
    fallback_bin: int,
) -> Tuple[Optional[int], str]:
    """
    Classify a lower-is-better metric according to ordered JSON rules.

    Example:

        [
            {"bin": 1, "max": 0.0},
            {"bin": 2, "max": 0.1},
            {"bin": 3, "max": 1.0}
        ]
    """
    if not is_finite_number(value):
        return None, "{} is missing or non-finite".format(
            metric_name
        )

    if not isinstance(rules, list) or len(rules) == 0:
        return None, "{} has no bin rules".format(
            metric_name
        )

    previous_maximum = -np.inf

    for rule in rules:
        if not isinstance(rule, dict):
            return None, "{} contains an invalid bin rule".format(
                metric_name
            )

        bin_number = rule.get("bin")
        maximum = rule.get("max")

        if bin_number is None or maximum is None:
            return None, (
                "{} has an unconfigured bin threshold"
            ).format(metric_name)

        try:
            bin_number = int(bin_number)
            maximum = float(maximum)
        except (TypeError, ValueError):
            return None, (
                "{} contains a non-numeric bin or threshold"
            ).format(metric_name)

        if maximum < previous_maximum:
            return None, (
                "{} bin thresholds are not monotonically increasing"
            ).format(metric_name)

        previous_maximum = maximum

        if float(value) <= maximum:
            return bin_number, ""

    return int(fallback_bin), ""


def classify_general_quality(
    metrics: Dict[str, Any],
    general_config: Dict[str, Any],
) -> Tuple[Optional[int], str]:
    """
    Assign the configurable single-surface general-quality bin.

    Euler mismatch is treated as the configured failure bin. For the other
    general metrics, the worst individual metric bin determines the overall
    bin.
    """
    fallback_bin = int(
        general_config.get("fallback_bin", 4)
    )

    missing_policy = general_config.get(
        "missing_metric_policy",
        "unscorable",
    )

    euler_config = general_config.get("euler", {})
    euler_target = euler_config.get("target", 2)
    euler_tolerance = float(
        euler_config.get("tolerance", 0)
    )
    euler_mismatch_bin = int(
        euler_config.get(
            "mismatch_bin",
            fallback_bin,
        )
    )

    euler_value = metrics.get("EulerCharacteristic")

    if not is_finite_number(euler_value):
        if missing_policy == "failure":
            return (
                fallback_bin,
                "Euler characteristic is unavailable",
            )

        return (
            None,
            "Euler characteristic is unavailable",
        )

    if abs(float(euler_value) - float(euler_target)) > euler_tolerance:
        return (
            euler_mismatch_bin,
            "EulerCharacteristic={} does not match target {}"
            .format(
                euler_value,
                euler_target,
            ),
        )

    metric_bin_rules = general_config.get(
        "metric_bins",
        {},
    )

    required_metrics = general_config.get(
        "required_metrics",
        [
            "SelfIntersectingFacesPct",
            "NormalConsistencyMeanDeg",
            "OrientationInconsistentFacesPct",
        ],
    )

    individual_bins = []
    reasons = []

    for metric_name in required_metrics:
        if metric_name not in metric_bin_rules:
            return (
                None,
                "No bin configuration for {}".format(
                    metric_name
                ),
            )

        metric_bin, metric_reason = (
            classify_lower_is_better_metric(
                metric_name=metric_name,
                value=metrics.get(metric_name),
                rules=metric_bin_rules[metric_name],
                fallback_bin=fallback_bin,
            )
        )

        if metric_bin is None:
            if missing_policy == "failure":
                individual_bins.append(fallback_bin)
                reasons.append(metric_reason)
                continue

            return None, metric_reason

        individual_bins.append(metric_bin)

    if len(individual_bins) == 0:
        return None, "No general-quality metrics were configured"

    overall_bin = max(individual_bins)

    if len(reasons) > 0:
        return overall_bin, "; ".join(reasons)

    return overall_bin, ""


def assign_score_bin(
    score: float,
    score_config: Dict[str, Any],
) -> Tuple[Optional[int], str]:
    rules = score_config.get("score_bins", [])
    above_maximum_bin = score_config.get(
        "above_maximum_bin",
        4,
    )

    if not is_finite_number(score):
        return None, "Composite score is not finite"

    if not isinstance(rules, list) or len(rules) == 0:
        return None, "Composite score bins are not configured"

    previous_maximum = -np.inf

    for rule in rules:
        bin_number = rule.get("bin")
        maximum = rule.get("max")

        if bin_number is None or maximum is None:
            return None, "Composite score bin contains null values"

        try:
            bin_number = int(bin_number)
            maximum = float(maximum)
        except (TypeError, ValueError):
            return None, "Composite score bins are non-numeric"

        if maximum < previous_maximum:
            return None, (
                "Composite score bins are not monotonically increasing"
            )

        previous_maximum = maximum

        if score <= maximum:
            return bin_number, ""

    return int(above_maximum_bin), ""


def safe_column_name(metric_name: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        metric_name,
    ).strip("_")


def calculate_composite_score(
    row: Dict[str, Any],
    score_config: Dict[str, Any],
) -> Tuple[Optional[float], Optional[int], str]:
    """
    Calculate the weighted composite z-score.

    Lower-is-better metric:

        z = (m - mean) / std

    Higher-is-better metric:

        z = (mean - m) / std

    z_score_mode can be:

    - signed:
        Better-than-reference metrics can have negative z-scores.

    - excess_only:
        z is replaced by max(0, z), so better-than-reference metrics cannot
        compensate for worse metrics.
    """
    metric_configs = score_config.get("metrics", {})

    enabled_metrics = [
        metric_name
        for metric_name, metric_config
        in metric_configs.items()
        if bool(metric_config.get("enabled", False))
    ]

    if len(enabled_metrics) == 0:
        return None, None, (
            "No composite-score metrics are enabled"
        )

    errors = []
    prepared = []
    weight_sum = 0.0

    for metric_name in enabled_metrics:
        metric_config = metric_configs[metric_name]

        metric_value = row.get(metric_name)
        reference_mean = metric_config.get("mean")
        reference_std = metric_config.get("std")
        weight = metric_config.get("weight")
        direction = metric_config.get(
            "direction",
            "lower",
        )

        if not is_finite_number(metric_value):
            errors.append(
                "{} is unavailable".format(metric_name)
            )
            continue

        if reference_mean is None:
            errors.append(
                "{} reference mean is null".format(
                    metric_name
                )
            )
            continue

        if reference_std is None:
            errors.append(
                "{} reference std is null".format(
                    metric_name
                )
            )
            continue

        if weight is None:
            errors.append(
                "{} weight is null".format(metric_name)
            )
            continue

        try:
            reference_mean = float(reference_mean)
            reference_std = float(reference_std)
            weight = float(weight)
        except (TypeError, ValueError):
            errors.append(
                "{} contains a non-numeric reference parameter"
                .format(metric_name)
            )
            continue

        if reference_std <= 0:
            errors.append(
                "{} reference std must be greater than zero"
                .format(metric_name)
            )
            continue

        if weight < 0:
            errors.append(
                "{} weight must not be negative".format(
                    metric_name
                )
            )
            continue

        if direction not in ("lower", "higher"):
            errors.append(
                "{} direction must be lower or higher"
                .format(metric_name)
            )
            continue

        prepared.append(
            (
                metric_name,
                float(metric_value),
                reference_mean,
                reference_std,
                weight,
                direction,
            )
        )

        weight_sum += weight

    if len(errors) > 0:
        return None, None, "; ".join(errors)

    if bool(
        score_config.get(
            "require_weights_sum_to_one",
            True,
        )
    ):
        tolerance = float(
            score_config.get(
                "weight_sum_tolerance",
                1e-6,
            )
        )

        if abs(weight_sum - 1.0) > tolerance:
            return (
                None,
                None,
                "Enabled metric weights sum to {}, not 1"
                .format(weight_sum),
            )

    z_score_mode = score_config.get(
        "z_score_mode",
        "signed",
    )

    if z_score_mode not in ("signed", "excess_only"):
        return (
            None,
            None,
            "Unknown z_score_mode: {}".format(
                z_score_mode
            ),
        )

    composite_score = 0.0

    for (
        metric_name,
        metric_value,
        reference_mean,
        reference_std,
        weight,
        direction,
    ) in prepared:
        if direction == "lower":
            z_score = (
                metric_value - reference_mean
            ) / reference_std
        else:
            z_score = (
                reference_mean - metric_value
            ) / reference_std

        if z_score_mode == "excess_only":
            z_score = max(0.0, z_score)

        contribution = weight * z_score

        safe_name = safe_column_name(metric_name)

        row["Z_{}".format(safe_name)] = float(
            z_score
        )
        row[
            "WeightedContribution_{}".format(
                safe_name
            )
        ] = float(contribution)

        composite_score += contribution

    score_bin, bin_error = assign_score_bin(
        composite_score,
        score_config,
    )

    if bin_error:
        return composite_score, None, bin_error

    return composite_score, score_bin, ""


def load_volume_file(
    path: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a NIfTI, NPY, or NPZ volume.

    Returns:
        data, affine

    affine is None for NPY/NPZ files.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Volume file does not exist: {}".format(path)
        )

    lower_path = path.lower()

    if lower_path.endswith(".nii") or lower_path.endswith(".nii.gz"):
        if nib is None:
            raise RuntimeError(
                "nibabel is required to read NIfTI files."
            )

        image = nib.load(path)
        data = np.asanyarray(image.dataobj)
        affine = np.asarray(
            image.affine,
            dtype=np.float64,
        )
    elif lower_path.endswith(".npy"):
        data = np.load(path)
        affine = None
    elif lower_path.endswith(".npz"):
        archive = np.load(path)

        if len(archive.files) == 0:
            raise ValueError(
                "NPZ file contains no arrays: {}".format(path)
            )

        data = archive[archive.files[0]]
        affine = None
    else:
        raise ValueError(
            "Unsupported volume format for {}. "
            "Use NIfTI, NPY, or NPZ.".format(path)
        )

    data = np.asarray(data)

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(
            "Volume must be three-dimensional; got {} for {}."
            .format(
                data.shape,
                path,
            )
        )

    return data, affine


def make_binary_mask(
    data: np.ndarray,
    label: Optional[float],
) -> np.ndarray:
    if label is None:
        return np.isfinite(data) & (data != 0)

    return np.isfinite(data) & np.isclose(
        data,
        float(label),
    )


def calculate_overlap_from_files(
    pred_seg_path: str,
    gt_seg_path: str,
    pred_label: Optional[float],
    gt_label: Optional[float],
    reference_image_path: Optional[str],
    require_reference_image: bool,
    affine_tolerance: float,
) -> Dict[str, Any]:
    """
    Read optional segmentation files and calculate Dice and IoU.

    Surface-to-volume voxelization is intentionally not performed here.
    If overlap metrics are required, both segmentations must already be
    represented in the same grid.
    """
    if require_reference_image and not reference_image_path:
        raise ValueError(
            "A reference image is required by the overlap configuration."
        )

    pred_data, pred_affine = load_volume_file(
        pred_seg_path
    )
    gt_data, gt_affine = load_volume_file(
        gt_seg_path
    )

    if pred_data.shape != gt_data.shape:
        raise ValueError(
            "Prediction and GT segmentations have different shapes: "
            "{} and {}.".format(
                pred_data.shape,
                gt_data.shape,
            )
        )

    if pred_affine is not None and gt_affine is not None:
        if not np.allclose(
            pred_affine,
            gt_affine,
            atol=affine_tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                "Prediction and GT NIfTI affines do not match."
            )

    if reference_image_path:
        reference_data, reference_affine = load_volume_file(
            reference_image_path
        )

        if reference_data.shape != pred_data.shape:
            raise ValueError(
                "Reference image shape {} does not match "
                "segmentation shape {}.".format(
                    reference_data.shape,
                    pred_data.shape,
                )
            )

        if reference_affine is not None:
            if pred_affine is not None and not np.allclose(
                pred_affine,
                reference_affine,
                atol=affine_tolerance,
                rtol=0.0,
            ):
                raise ValueError(
                    "Prediction segmentation affine does not match "
                    "the reference image."
                )

            if gt_affine is not None and not np.allclose(
                gt_affine,
                reference_affine,
                atol=affine_tolerance,
                rtol=0.0,
            ):
                raise ValueError(
                    "GT segmentation affine does not match "
                    "the reference image."
                )

    pred_mask = make_binary_mask(
        pred_data,
        pred_label,
    )

    gt_mask = make_binary_mask(
        gt_data,
        gt_label,
    )

    return compute_binary_overlap(
        pred_mask,
        gt_mask,
    )


def initialize_output_row(
    args: argparse.Namespace,
    profile_name: str,
) -> Dict[str, Any]:
    row = {
        "SchemaVersion": SCHEMA_VERSION,
        "EvaluationMode": args.mode,
        "ConfigPath": os.path.abspath(args.config),
        "ConfigProfile": profile_name,
        "Subject": args.subject or "",
        "Session": args.session or "",
        "Hemisphere": args.hemisphere or "",
        "Surface": args.surface or "",
        "Label": args.label or "",
        "GTPath": os.path.abspath(args.gt),
        "PredPath": (
            os.path.abspath(args.pred)
            if args.pred
            else ""
        ),
        "PredSegPath": (
            os.path.abspath(args.pred_seg)
            if args.pred_seg
            else ""
        ),
        "GTSegPath": (
            os.path.abspath(args.gt_seg)
            if args.gt_seg
            else ""
        ),
        "ReferenceImagePath": (
            os.path.abspath(args.reference_image)
            if args.reference_image
            else ""
        ),
    }

    for prefix in ("GT_", "Pred_"):
        for key in GENERAL_METRIC_KEYS:
            row["{}{}".format(prefix, key)] = np.nan

        row[
            "{}GeneralQualityBin".format(prefix)
        ] = np.nan

        row[
            "{}GeneralQualityReason".format(prefix)
        ] = ""

    row["GTEligible"] = None
    row["PredHardFailure"] = None

    for key in SURFACE_DISTANCE_METRIC_KEYS:
        row[key] = np.nan

    row["DistanceMetricsValid"] = False
    row["DistanceMetricError"] = ""

    row["RMSD"] = np.nan
    row["RMSDValid"] = False
    row["RMSDError"] = ""

    row["Dice"] = np.nan
    row["IoU"] = np.nan
    row["OverlapIntersectionVoxelCount"] = np.nan
    row["OverlapUnionVoxelCount"] = np.nan
    row["PredForegroundVoxelCount"] = np.nan
    row["GTForegroundVoxelCount"] = np.nan
    row["OverlapStatus"] = "NOT_REQUESTED"
    row["OverlapError"] = ""

    row["SpecificMetricsStatus"] = "NOT_RUN"

    row["CompositeScore"] = np.nan
    row["ScoreQualityBin"] = np.nan
    row["FinalQualityBin"] = np.nan
    row["CompositeScoreStatus"] = "NOT_RUN"
    row["CompositeScoreError"] = ""
    row["EvaluationStatus"] = "NOT_COMPLETED"
    row["FailureReasons"] = ""

    return row


def run_evaluation(
    args: argparse.Namespace,
) -> pd.DataFrame:
    raw_config = load_config(args.config)

    config, profile_name = resolve_config(
        raw_config,
        hemisphere=args.hemisphere or "",
        surface=args.surface or "",
    )

    row = initialize_output_row(
        args,
        profile_name,
    )

    score_config = config.get(
        "composite_score",
        {},
    )

    # Create stable z-score/contribution columns even when scoring is not
    # possible yet because reference parameters are null.
    for metric_name in score_config.get(
        "metrics",
        {},
    ):
        safe_name = safe_column_name(metric_name)
        row["Z_{}".format(safe_name)] = np.nan
        row[
            "WeightedContribution_{}".format(
                safe_name
            )
        ] = np.nan

    general_config = config.get(
        "general_quality",
        {},
    )

    calculate_self_intersections = bool(
        general_config.get(
            "calculate_self_intersections",
            True,
        )
    )

    gt_mesh = load_mesh_unprocessed(args.gt)

    gt_metrics = compute_general_mesh_metrics(
        gt_mesh.vertices,
        gt_mesh.faces,
        calculate_self_intersections=calculate_self_intersections,
    )

    row.update(prefix_metrics(gt_metrics, "GT_"))

    gt_bin, gt_bin_reason = classify_general_quality(
        gt_metrics,
        general_config,
    )

    row["GT_GeneralQualityBin"] = (
        gt_bin if gt_bin is not None else np.nan
    )
    row["GT_GeneralQualityReason"] = gt_bin_reason

    gt_eligible_bins = [
        int(item)
        for item in general_config.get(
            "gt_eligible_bins",
            [1, 2],
        )
    ]

    if gt_bin is None:
        row["GTEligible"] = None
    else:
        row["GTEligible"] = (
            int(gt_bin) in gt_eligible_bins
        )

    if args.mode == "gt-only":
        if gt_bin is None:
            row["EvaluationStatus"] = (
                "GT_ONLY_UNSCORABLE"
            )
            row["FailureReasons"] = gt_bin_reason
        elif bool(row["GTEligible"]):
            row["EvaluationStatus"] = (
                "GT_ONLY_ELIGIBLE"
            )
        else:
            row["EvaluationStatus"] = (
                "GT_ONLY_EXCLUDED"
            )
            row["FailureReasons"] = (
                gt_bin_reason
                or "GT quality bin is not eligible"
            )

        return pd.DataFrame([row])

    if not args.pred:
        raise ValueError(
            "--pred is required in pair mode."
        )

    pred_mesh = load_mesh_unprocessed(args.pred)

    pred_metrics = compute_general_mesh_metrics(
        pred_mesh.vertices,
        pred_mesh.faces,
        calculate_self_intersections=calculate_self_intersections,
    )

    row.update(prefix_metrics(pred_metrics, "Pred_"))

    pred_bin, pred_bin_reason = classify_general_quality(
        pred_metrics,
        general_config,
    )

    row["Pred_GeneralQualityBin"] = (
        pred_bin if pred_bin is not None else np.nan
    )
    row["Pred_GeneralQualityReason"] = pred_bin_reason

    hard_failure_bins = [
        int(item)
        for item in general_config.get(
            "prediction_hard_failure_bins",
            [4],
        )
    ]

    if pred_bin is None:
        row["PredHardFailure"] = None
    else:
        row["PredHardFailure"] = (
            int(pred_bin) in hard_failure_bins
        )

    evaluation_config = config.get(
        "evaluation",
        {},
    )

    compute_specific_even_if_screening_fails = bool(
        evaluation_config.get(
            "compute_specific_metrics_even_if_screening_fails",
            True,
        )
    )

    screening_passed = (
        gt_bin is not None
        and bool(row["GTEligible"])
        and pred_bin is not None
        and not bool(row["PredHardFailure"])
    )

    should_compute_specific = (
        screening_passed
        or compute_specific_even_if_screening_fails
    )

    if should_compute_specific:
        surface_distance_config = config.get(
            "surface_distance",
            {},
        )

        n_points = (
            args.n_points
            if args.n_points is not None
            else int(
                surface_distance_config.get(
                    "n_points",
                    10000,
                )
            )
        )

        seed = (
            args.seed
            if args.seed is not None
            else int(
                surface_distance_config.get(
                    "seed",
                    42,
                )
            )
        )

        device_name = (
            args.device
            if args.device is not None
            else surface_distance_config.get(
                "device",
                (
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                ),
            )
        )

        if (
            device_name == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() "
                "is false."
            )

        try:
            distance_metrics = (
                compute_surface_distance_metrics(
                    pred_vertices=pred_mesh.vertices,
                    pred_faces=pred_mesh.faces,
                    gt_vertices=gt_mesh.vertices,
                    gt_faces=gt_mesh.faces,
                    n_points=n_points,
                    device=device_name,
                    seed=seed,
                )
            )

            row.update(distance_metrics)
            row["DistanceMetricsValid"] = True
        except Exception as exc:
            row["DistanceMetricsValid"] = False
            row["DistanceMetricError"] = (
                "{}: {}".format(
                    type(exc).__name__,
                    str(exc),
                )
            )

        rmsd_config = config.get("rmsd", {})

        if bool(rmsd_config.get("enabled", True)):
            rmsd_result = compute_correspondence_rmsd(
                pred_vertices=pred_mesh.vertices,
                gt_vertices=gt_mesh.vertices,
                pred_faces=pred_mesh.faces,
                gt_faces=gt_mesh.faces,
                require_identical_faces=bool(
                    rmsd_config.get(
                        "require_identical_faces",
                        False,
                    )
                ),
            )

            row.update(rmsd_result)
        else:
            row["RMSDError"] = (
                "RMSD calculation disabled by configuration"
            )

        overlap_config = config.get("overlap", {})
        overlap_mode = overlap_config.get(
            "mode",
            "disabled",
        )

        if overlap_mode == "disabled":
            row["OverlapStatus"] = "DISABLED"

        elif overlap_mode == "segmentation_files":
            if not args.pred_seg or not args.gt_seg:
                row["OverlapStatus"] = (
                    "MISSING_INPUTS"
                )
                row["OverlapError"] = (
                    "segmentation_files mode requires both "
                    "--pred_seg and --gt_seg"
                )
            else:
                try:
                    overlap_metrics = (
                        calculate_overlap_from_files(
                            pred_seg_path=args.pred_seg,
                            gt_seg_path=args.gt_seg,
                            pred_label=args.pred_seg_label,
                            gt_label=args.gt_seg_label,
                            reference_image_path=(
                                args.reference_image
                            ),
                            require_reference_image=bool(
                                overlap_config.get(
                                    "require_reference_image",
                                    False,
                                )
                            ),
                            affine_tolerance=float(
                                overlap_config.get(
                                    "affine_tolerance",
                                    1e-5,
                                )
                            ),
                        )
                    )

                    row.update(overlap_metrics)
                    row["OverlapStatus"] = "VALID"
                except Exception as exc:
                    row["OverlapStatus"] = "ERROR"
                    row["OverlapError"] = (
                        "{}: {}".format(
                            type(exc).__name__,
                            str(exc),
                        )
                    )

        elif overlap_mode == "surface_voxelization":
            row["OverlapStatus"] = "NOT_IMPLEMENTED"
            row["OverlapError"] = (
                "surface_voxelization is reserved for a future "
                "implementation. Use segmentation_files mode with "
                "prediction and GT masks in a shared grid."
            )

        else:
            row["OverlapStatus"] = "ERROR"
            row["OverlapError"] = (
                "Unknown overlap mode: {}".format(
                    overlap_mode
                )
            )

        row["SpecificMetricsStatus"] = "COMPUTED"

    else:
        row["SpecificMetricsStatus"] = (
            "SKIPPED_BY_SCREENING"
        )

    # Final framework decision.
    if gt_bin is None:
        row["EvaluationStatus"] = "GT_QA_UNSCORABLE"
        row["FailureReasons"] = gt_bin_reason
        return pd.DataFrame([row])

    if not bool(row["GTEligible"]):
        row["EvaluationStatus"] = "GT_EXCLUDED"
        row["FailureReasons"] = (
            gt_bin_reason
            or "GT quality bin is not eligible"
        )
        return pd.DataFrame([row])

    if pred_bin is None:
        row["EvaluationStatus"] = (
            "PRED_SCREENING_UNSCORABLE"
        )
        row["FailureReasons"] = pred_bin_reason
        return pd.DataFrame([row])

    if bool(row["PredHardFailure"]):
        failure_bin = int(
            general_config.get(
                "fallback_bin",
                4,
            )
        )

        row["FinalQualityBin"] = failure_bin
        row["CompositeScoreStatus"] = (
            "NOT_REQUIRED_HARD_FAILURE"
        )
        row["EvaluationStatus"] = "PRED_HARD_FAILURE"
        row["FailureReasons"] = (
            pred_bin_reason
            or "Prediction failed general-quality screening"
        )
        return pd.DataFrame([row])

    composite_score, score_bin, score_error = (
        calculate_composite_score(
            row,
            score_config,
        )
    )

    if composite_score is not None:
        row["CompositeScore"] = composite_score

    if score_bin is not None:
        row["ScoreQualityBin"] = score_bin
        row["FinalQualityBin"] = score_bin
        row["CompositeScoreStatus"] = "VALID"
        row["EvaluationStatus"] = "EVALUATED"
    else:
        row["CompositeScoreStatus"] = "UNAVAILABLE"
        row["CompositeScoreError"] = score_error
        row["EvaluationStatus"] = (
            "COMPOSITE_SCORE_UNAVAILABLE"
        )
        row["FailureReasons"] = score_error

    return pd.DataFrame([row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a GT cortical surface alone or compare a "
            "reconstructed surface with GT."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["gt-only", "pair"],
        default="pair",
        help=(
            "gt-only computes Part 1 metrics only. "
            "pair computes Parts 1 and 2."
        ),
    )

    parser.add_argument(
        "--gt_only",
        action="store_true",
        help=(
            "Compatibility alias for --mode gt-only."
        ),
    )

    parser.add_argument(
        "--gt",
        required=True,
        help="Path to the ground-truth mesh.",
    )

    parser.add_argument(
        "--pred",
        default=None,
        help=(
            "Path to the predicted mesh. Required in pair mode."
        ),
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to quality_config.json.",
    )

    parser.add_argument(
        "--subject",
        default="",
        help="Subject identifier stored in the output.",
    )

    parser.add_argument(
        "--session",
        default="",
        help="Session identifier stored in the output.",
    )

    parser.add_argument(
        "--hemisphere",
        default="",
        help=(
            "Hemisphere identifier, for example L or R."
        ),
    )

    parser.add_argument(
        "--surface",
        default="",
        help=(
            "Surface identifier, for example pial or white."
        ),
    )

    parser.add_argument(
        "--label",
        default=None,
        help="Optional combined surface label.",
    )

    parser.add_argument(
        "--n_points",
        type=int,
        default=None,
        help=(
            "Override the number of sampled points specified "
            "in the JSON configuration."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the random seed specified in the JSON."
        ),
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help=(
            "Override the distance-calculation device specified "
            "in the JSON."
        ),
    )

    parser.add_argument(
        "--pred_seg",
        default=None,
        help=(
            "Optional predicted segmentation in NIfTI, NPY, or "
            "NPZ format."
        ),
    )

    parser.add_argument(
        "--gt_seg",
        default=None,
        help=(
            "Optional GT segmentation in NIfTI, NPY, or NPZ format."
        ),
    )

    parser.add_argument(
        "--reference_image",
        default=None,
        help=(
            "Optional reference image defining the expected "
            "segmentation grid."
        ),
    )

    parser.add_argument(
        "--pred_seg_label",
        type=float,
        default=None,
        help=(
            "Predicted segmentation label. If omitted, all "
            "nonzero voxels are treated as foreground."
        ),
    )

    parser.add_argument(
        "--gt_seg_label",
        type=float,
        default=None,
        help=(
            "GT segmentation label. If omitted, all nonzero "
            "voxels are treated as foreground."
        ),
    )

    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional output CSV path.",
    )

    parser.add_argument(
        "--out_json",
        default=None,
        help="Optional output JSON path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gt_only:
        args.mode = "gt-only"

    if args.mode == "pair" and not args.pred:
        raise ValueError(
            "--pred must be supplied in pair mode."
        )

    if args.label is None:
        if (
            args.subject
            or args.session
            or args.hemisphere
            or args.surface
        ):
            args.label = "_".join(
                item
                for item in [
                    args.subject,
                    args.session,
                    args.hemisphere,
                    args.surface,
                ]
                if item
            )
        else:
            args.label = os.path.basename(args.gt)

    dataframe = run_evaluation(args)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print(dataframe.to_string(index=False))

    if args.out_csv:
        output_directory = os.path.dirname(
            os.path.abspath(args.out_csv)
        )

        os.makedirs(
            output_directory,
            exist_ok=True,
        )

        dataframe.to_csv(
            args.out_csv,
            index=False,
        )

    if args.out_json:
        output_directory = os.path.dirname(
            os.path.abspath(args.out_json)
        )

        os.makedirs(
            output_directory,
            exist_ok=True,
        )

        # Pandas converts NaN to JSON null.
        records = json.loads(
            dataframe.to_json(
                orient="records"
            )
        )

        with open(args.out_json, "w") as file_handle:
            json.dump(
                records,
                file_handle,
                indent=2,
            )


if __name__ == "__main__":
    main()