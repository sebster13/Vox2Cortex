"""
Evaluation metrics for cortical surface reconstruction.

This module contains:

General single-surface metrics
------------------------------
- Euler characteristic
- Mesh edge/topology diagnostics
- Percentage of self-intersecting faces
- Adjacent-face normal consistency
- Edge-winding orientation consistency
- Signed enclosed volume as a diagnostic

Prediction-versus-ground-truth metrics
--------------------------------------
- ASSD
- Hausdorff distance
- HD90, HD95, HD99
- Unsquared Chamfer distance
- Squared Chamfer distance
- Symmetric point-to-surface RMSE
- Correspondence-based vertex RMSD
- Binary Dice and IoU

The legacy EvalMetric interface is retained for compatibility with the
rest of the project.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import pymeshlab as pyml
except ImportError:
    pyml = None

from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds


point_face_distance = _PointFaceDistance.apply


GENERAL_METRIC_KEYS = [
    "VertexCount",
    "FaceCount",
    "UniqueEdgeCount",
    "EulerCharacteristic",
    "BoundaryEdgeCount",
    "NonManifoldEdgeCount",
    "DegenerateFaceCount",
    "DuplicateFaceCount",
    "UnreferencedVertexCount",
    "SelfIntersectingFaceCount",
    "SelfIntersectingFacesPct",
    "NormalConsistencyMeanDeg",
    "NormalConsistencyP95Deg",
    "NormalConsistencyAdjacentPairCount",
    "OrientationInconsistentFaceCount",
    "OrientationInconsistentFacesPct",
    "OrientationConstraintConflictCount",
    "FaceConnectedComponentCount",
    "SignedVolume",
    "AbsoluteSignedVolume",
    "GeneralMetricErrors",
]


SURFACE_DISTANCE_METRIC_KEYS = [
    "ASSD",
    "HD",
    "HD90",
    "HD95",
    "HD99",
    "Chamfer",
    "ChamferSquared",
    "SurfaceRMSE",
    "NumSurfaceSamples",
]


def _validate_mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and standardize NumPy mesh arrays.

    No mesh processing, vertex merging, or topology repair is performed.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            "Vertices must have shape (V, 3); got {}.".format(vertices.shape)
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            "Faces must have shape (F, 3); got {}. "
            "Only triangular meshes are supported.".format(faces.shape)
        )

    if vertices.shape[0] == 0:
        raise ValueError("Mesh contains no vertices.")

    if faces.shape[0] == 0:
        raise ValueError("Mesh contains no faces.")

    if not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices contain NaN or infinite values.")

    if faces.min() < 0 or faces.max() >= vertices.shape[0]:
        raise ValueError(
            "Mesh contains face indices outside the valid vertex range."
        )

    return vertices, faces


def _build_edge_incidence(faces: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Construct unique undirected edges and face-edge incidence information.

    The directed edge order of each triangle is preserved separately so that
    neighboring-face winding consistency can be evaluated.
    """
    n_faces = faces.shape[0]

    directed_edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )

    face_ids = np.tile(np.arange(n_faces, dtype=np.int64), 3)

    undirected_edges = np.sort(directed_edges, axis=1)

    unique_edges, inverse, counts = np.unique(
        undirected_edges,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    # +1 means the directed edge traverses from the smaller to the larger
    # vertex index. -1 means the reverse.
    directions = np.where(
        directed_edges[:, 0] < directed_edges[:, 1],
        1,
        -1,
    ).astype(np.int8)

    # Degenerate edges such as (v, v) have no meaningful direction.
    directions[
        directed_edges[:, 0] == directed_edges[:, 1]
    ] = 0

    order = np.argsort(inverse, kind="mergesort")
    offsets = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum(counts, dtype=np.int64),
        ]
    )

    return {
        "directed_edges": directed_edges,
        "undirected_edges": undirected_edges,
        "unique_edges": unique_edges,
        "inverse": inverse,
        "counts": counts,
        "directions": directions,
        "face_ids": face_ids,
        "order": order,
        "offsets": offsets,
    }


def _edge_group_indices(
    edge_data: Dict[str, np.ndarray],
    edge_id: int,
) -> np.ndarray:
    start = edge_data["offsets"][edge_id]
    end = edge_data["offsets"][edge_id + 1]
    return edge_data["order"][start:end]


def _compute_face_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate unit face normals and a validity mask.

    Degenerate faces receive a zero normal and are marked invalid.
    """
    triangles = vertices[faces]

    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )

    lengths = np.linalg.norm(cross, axis=1)

    coordinate_extent = np.ptp(vertices, axis=0)
    scale = max(float(np.linalg.norm(coordinate_extent)), 1.0)
    epsilon = np.finfo(np.float64).eps * (scale ** 2) * 100.0

    valid = lengths > epsilon

    normals = np.zeros_like(cross, dtype=np.float64)
    normals[valid] = cross[valid] / lengths[valid, None]

    return normals, valid, epsilon


def compute_mesh_topology_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    edge_data: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Calculate Euler characteristic and supporting mesh diagnostics.

    The Euler characteristic is calculated from the original mesh arrays:

        chi = V - E + F

    where E is the number of unique undirected edges.
    """
    vertices, faces = _validate_mesh_arrays(vertices, faces)

    if edge_data is None:
        edge_data = _build_edge_incidence(faces)

    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0])
    n_edges = int(edge_data["unique_edges"].shape[0])

    euler_characteristic = n_vertices - n_edges + n_faces

    boundary_edges = int(np.count_nonzero(edge_data["counts"] == 1))
    non_manifold_edges = int(np.count_nonzero(edge_data["counts"] > 2))

    _, valid_faces, _ = _compute_face_normals(vertices, faces)
    degenerate_faces = int(np.count_nonzero(~valid_faces))

    canonical_faces = np.sort(faces, axis=1)
    unique_face_count = int(
        np.unique(canonical_faces, axis=0).shape[0]
    )
    duplicate_faces = n_faces - unique_face_count

    referenced_vertices = np.unique(faces.reshape(-1))
    unreferenced_vertices = n_vertices - int(referenced_vertices.shape[0])

    return {
        "VertexCount": n_vertices,
        "FaceCount": n_faces,
        "UniqueEdgeCount": n_edges,
        "EulerCharacteristic": int(euler_characteristic),
        "BoundaryEdgeCount": boundary_edges,
        "NonManifoldEdgeCount": non_manifold_edges,
        "DegenerateFaceCount": degenerate_faces,
        "DuplicateFaceCount": duplicate_faces,
        "UnreferencedVertexCount": unreferenced_vertices,
    }


def compute_normal_consistency(
    vertices: np.ndarray,
    faces: np.ndarray,
    edge_data: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Calculate adjacent-face normal angles in degrees.

    For each manifold edge shared by exactly two distinct faces, the angle is:

        theta = arccos(clip(n_i dot n_j, -1, 1))

    The principal output is the mean angle in degrees. Lower values indicate
    smoother and more locally consistent face normals.

    Degenerate faces and non-manifold edges are excluded from the angle
    calculation and are reported separately by topology metrics.
    """
    vertices, faces = _validate_mesh_arrays(vertices, faces)

    if edge_data is None:
        edge_data = _build_edge_incidence(faces)

    normals, valid_faces, _ = _compute_face_normals(vertices, faces)

    angles = []

    for edge_id, count in enumerate(edge_data["counts"]):
        if count != 2:
            continue

        incidence_indices = _edge_group_indices(edge_data, edge_id)
        incident_faces = edge_data["face_ids"][incidence_indices]

        face_a = int(incident_faces[0])
        face_b = int(incident_faces[1])

        if face_a == face_b:
            continue

        if not valid_faces[face_a] or not valid_faces[face_b]:
            continue

        dot_product = float(
            np.dot(normals[face_a], normals[face_b])
        )
        dot_product = float(np.clip(dot_product, -1.0, 1.0))

        angle_deg = float(np.degrees(np.arccos(dot_product)))
        angles.append(angle_deg)

    if len(angles) == 0:
        return {
            "NormalConsistencyMeanDeg": np.nan,
            "NormalConsistencyP95Deg": np.nan,
            "NormalConsistencyAdjacentPairCount": 0,
        }

    angle_array = np.asarray(angles, dtype=np.float64)

    return {
        "NormalConsistencyMeanDeg": float(np.mean(angle_array)),
        "NormalConsistencyP95Deg": float(
            np.percentile(angle_array, 95)
        ),
        "NormalConsistencyAdjacentPairCount": int(
            angle_array.shape[0]
        ),
    }


def compute_orientation_consistency(
    vertices: np.ndarray,
    faces: np.ndarray,
    edge_data: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Evaluate local face-winding consistency through shared mesh edges.

    Two consistently wound neighboring triangles traverse their common edge
    in opposite directions.

    A parity graph is constructed over adjacent faces. For each connected
    component, the method determines the smaller set of faces that would need
    to be flipped to make the component locally consistent.

    Non-manifold edges and parity conflicts are treated as orientation
    inconsistencies.

    This method is preferred over checking per-face signed tetrahedron signs,
    because per-face signed volume signs are not reliable for highly concave
    cortical surfaces.
    """
    vertices, faces = _validate_mesh_arrays(vertices, faces)

    if edge_data is None:
        edge_data = _build_edge_incidence(faces)

    n_faces = int(faces.shape[0])

    # Each adjacency entry is:
    #   (neighbor_face, required_xor)
    #
    # required_xor = 0 means the two original directed shared-edge
    # orientations are already opposite.
    #
    # required_xor = 1 means exactly one of the two faces must be flipped.
    adjacency = [[] for _ in range(n_faces)]

    non_manifold_faces = set()

    for edge_id, count in enumerate(edge_data["counts"]):
        incidence_indices = _edge_group_indices(edge_data, edge_id)

        incident_faces = edge_data["face_ids"][incidence_indices]
        directions = edge_data["directions"][incidence_indices]

        if count == 2:
            face_a = int(incident_faces[0])
            face_b = int(incident_faces[1])
            dir_a = int(directions[0])
            dir_b = int(directions[1])

            if face_a == face_b or dir_a == 0 or dir_b == 0:
                non_manifold_faces.add(face_a)
                non_manifold_faces.add(face_b)
                continue

            required_xor = 1 if dir_a == dir_b else 0

            adjacency[face_a].append((face_b, required_xor))
            adjacency[face_b].append((face_a, required_xor))

        elif count > 2:
            for face_id in incident_faces:
                non_manifold_faces.add(int(face_id))

    parity = np.full(n_faces, -1, dtype=np.int8)
    components = []
    traversal_conflict_faces = set()

    for start_face in range(n_faces):
        if parity[start_face] != -1:
            continue

        parity[start_face] = 0
        queue = deque([start_face])
        component = []

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbor, required_xor in adjacency[current]:
                expected = parity[current] ^ required_xor

                if parity[neighbor] == -1:
                    parity[neighbor] = expected
                    queue.append(neighbor)
                elif parity[neighbor] != expected:
                    traversal_conflict_faces.add(current)
                    traversal_conflict_faces.add(neighbor)

        components.append(component)

    # Each connected component can be globally inverted without changing
    # its local consistency. Select the orientation requiring fewer flips.
    final_flip = np.zeros(n_faces, dtype=np.int8)

    for component in components:
        component_array = np.asarray(component, dtype=np.int64)
        number_one = int(np.count_nonzero(parity[component_array] == 1))
        number_zero = int(component_array.shape[0] - number_one)

        invert_component = number_one > number_zero

        if invert_component:
            final_flip[component_array] = (
                1 - parity[component_array]
            )
        else:
            final_flip[component_array] = parity[component_array]

    constraint_conflicts = 0
    constraint_conflict_faces = set()

    for face_a in range(n_faces):
        for face_b, required_xor in adjacency[face_a]:
            if face_a >= face_b:
                continue

            observed_xor = int(final_flip[face_a] ^ final_flip[face_b])

            if observed_xor != required_xor:
                constraint_conflicts += 1
                constraint_conflict_faces.add(face_a)
                constraint_conflict_faces.add(face_b)

    inconsistent_faces = set(
        np.flatnonzero(final_flip == 1).astype(int).tolist()
    )
    inconsistent_faces.update(non_manifold_faces)
    inconsistent_faces.update(traversal_conflict_faces)
    inconsistent_faces.update(constraint_conflict_faces)

    inconsistent_count = len(inconsistent_faces)
    inconsistent_pct = (
        100.0 * inconsistent_count / float(n_faces)
        if n_faces > 0
        else np.nan
    )

    return {
        "OrientationInconsistentFaceCount": int(
            inconsistent_count
        ),
        "OrientationInconsistentFacesPct": float(
            inconsistent_pct
        ),
        "OrientationConstraintConflictCount": int(
            constraint_conflicts
        ),
        "FaceConnectedComponentCount": int(len(components)),
    }


def compute_signed_volume(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate the global signed mesh volume.

    The mesh is centered before calculation to improve numerical stability.
    For a closed consistently wound mesh, the absolute value corresponds to
    enclosed volume.

    The sign depends on global winding and coordinate convention. It is
    reported as a diagnostic, not used as the local orientation-consistency
    metric.
    """
    vertices, faces = _validate_mesh_arrays(vertices, faces)

    center = np.mean(vertices, axis=0)
    centered = vertices - center

    triangles = centered[faces]

    signed_contributions = np.einsum(
        "ij,ij->i",
        triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2]),
    ) / 6.0

    signed_volume = float(np.sum(signed_contributions))

    return {
        "SignedVolume": signed_volume,
        "AbsoluteSignedVolume": abs(signed_volume),
    }


def compute_self_intersections(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate the number and percentage of faces involved in at least one
    self-intersection using PyMeshLab.
    """
    vertices, faces = _validate_mesh_arrays(vertices, faces)

    if pyml is None:
        raise RuntimeError(
            "pymeshlab is not installed; self-intersections cannot "
            "be calculated."
        )

    mesh_set = pyml.MeshSet()
    mesh_set.add_mesh(
        pyml.Mesh(
            vertex_matrix=np.asarray(vertices, dtype=np.float64),
            face_matrix=np.asarray(faces, dtype=np.int32),
        )
    )

    face_count = int(mesh_set.current_mesh().face_number())

    if face_count == 0:
        return {
            "SelfIntersectingFaceCount": 0,
            "SelfIntersectingFacesPct": 0.0,
        }

    selection_completed = False

    # Newer PyMeshLab API.
    try:
        mesh_set.compute_selection_by_self_intersections_per_face()
        selection_completed = True
    except (AttributeError, TypeError, RuntimeError):
        pass

    # Older PyMeshLab API used by the original project.
    if not selection_completed:
        try:
            mesh_set.select_self_intersecting_faces()
            selection_completed = True
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise RuntimeError(
                "The installed PyMeshLab version does not expose a "
                "supported self-intersection selection method."
            ) from exc

    try:
        selection = np.asarray(
            mesh_set.current_mesh().face_selection_array()
        ).astype(bool)
        selected_count = int(np.count_nonzero(selection))
    except (AttributeError, TypeError, RuntimeError):
        # Fallback for older versions without face_selection_array().
        before = int(mesh_set.current_mesh().face_number())

        deleted = False

        try:
            mesh_set.meshing_remove_selected_faces()
            deleted = True
        except (AttributeError, TypeError, RuntimeError):
            pass

        if not deleted:
            try:
                mesh_set.delete_selected_faces()
                deleted = True
            except (AttributeError, TypeError, RuntimeError) as exc:
                raise RuntimeError(
                    "Self-intersecting faces were selected, but the "
                    "installed PyMeshLab version could not expose or "
                    "delete the selected faces."
                ) from exc

        after = int(mesh_set.current_mesh().face_number())
        selected_count = before - after

    percentage = 100.0 * selected_count / float(face_count)

    return {
        "SelfIntersectingFaceCount": int(selected_count),
        "SelfIntersectingFacesPct": float(percentage),
    }


def compute_general_mesh_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    calculate_self_intersections: bool = True,
) -> Dict[str, Any]:
    """
    Calculate all general single-surface quality metrics.

    A failure of the optional PyMeshLab self-intersection calculation does not
    suppress the other metrics. Instead, the SIF values are returned as NaN
    and the error is recorded in GeneralMetricErrors.
    """
    vertices, faces = _validate_mesh_arrays(vertices, faces)
    edge_data = _build_edge_incidence(faces)

    results = {}
    errors = []

    results.update(
        compute_mesh_topology_metrics(
            vertices,
            faces,
            edge_data=edge_data,
        )
    )

    results.update(
        compute_normal_consistency(
            vertices,
            faces,
            edge_data=edge_data,
        )
    )

    results.update(
        compute_orientation_consistency(
            vertices,
            faces,
            edge_data=edge_data,
        )
    )

    results.update(compute_signed_volume(vertices, faces))

    if calculate_self_intersections:
        try:
            results.update(
                compute_self_intersections(vertices, faces)
            )
        except Exception as exc:
            results["SelfIntersectingFaceCount"] = np.nan
            results["SelfIntersectingFacesPct"] = np.nan
            errors.append(
                "SelfIntersections: {}: {}".format(
                    type(exc).__name__,
                    str(exc),
                )
            )
    else:
        results["SelfIntersectingFaceCount"] = np.nan
        results["SelfIntersectingFacesPct"] = np.nan
        errors.append("SelfIntersections: calculation disabled")

    results["GeneralMetricErrors"] = "; ".join(errors)

    # Ensure a stable set of output keys.
    for key in GENERAL_METRIC_KEYS:
        if key not in results:
            results[key] = np.nan

    return results


def _point_mesh_face_distance_unidirectional(
    pointcloud: Pointclouds,
    target_mesh: Meshes,
) -> torch.Tensor:
    """
    Calculate non-squared point-to-triangle distances.

    PyTorch3D's private _PointFaceDistance operation returns squared
    distances, so a square root is applied before returning.
    """
    points = pointcloud.points_packed()
    points_first_idx = pointcloud.cloud_to_packed_first_idx()
    max_points = pointcloud.num_points_per_cloud().max().item()

    verts_packed = target_mesh.verts_packed()
    faces_packed = target_mesh.faces_packed()
    triangles = verts_packed[faces_packed]
    triangles_first_idx = (
        target_mesh.mesh_to_faces_packed_first_idx()
    )

    squared_distance = point_face_distance(
        points,
        points_first_idx,
        triangles,
        triangles_first_idx,
        max_points,
    )

    return torch.sqrt(torch.clamp(squared_distance, min=0.0))


def _as_mesh_tensor(
    value: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)

    return torch.as_tensor(
        value,
        dtype=dtype,
        device=device,
    )


def _compute_surface_distance_metrics_torch(
    pred_vertices: Any,
    pred_faces: Any,
    gt_vertices: Any,
    gt_faces: Any,
    n_points: int,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    if n_points <= 0:
        raise ValueError("n_points must be greater than zero.")

    pred_vertices = _as_mesh_tensor(
        pred_vertices,
        torch.float32,
        device,
    )
    pred_faces = _as_mesh_tensor(
        pred_faces,
        torch.int64,
        device,
    )

    gt_vertices = _as_mesh_tensor(
        gt_vertices,
        torch.float32,
        device,
    )
    gt_faces = _as_mesh_tensor(
        gt_faces,
        torch.int64,
        device,
    )

    if pred_vertices.ndim != 2 or pred_vertices.shape[1] != 3:
        raise ValueError("Prediction vertices must have shape (V, 3).")

    if gt_vertices.ndim != 2 or gt_vertices.shape[1] != 3:
        raise ValueError("GT vertices must have shape (V, 3).")

    if pred_faces.ndim != 2 or pred_faces.shape[1] != 3:
        raise ValueError("Prediction faces must have shape (F, 3).")

    if gt_faces.ndim != 2 or gt_faces.shape[1] != 3:
        raise ValueError("GT faces must have shape (F, 3).")

    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    pred_mesh = Meshes(
        verts=[pred_vertices],
        faces=[pred_faces],
    )

    gt_mesh = Meshes(
        verts=[gt_vertices],
        faces=[gt_faces],
    )

    with torch.no_grad():
        pred_sampled = sample_points_from_meshes(
            pred_mesh,
            n_points,
        )
        gt_sampled = sample_points_from_meshes(
            gt_mesh,
            n_points,
        )

        pred_pointcloud = Pointclouds(pred_sampled)
        gt_pointcloud = Pointclouds(gt_sampled)

        # Prediction-source points to the complete GT triangle surface.
        pred_to_gt = _point_mesh_face_distance_unidirectional(
            pred_pointcloud,
            gt_mesh,
        )

        # GT-source points to the complete prediction triangle surface.
        gt_to_pred = _point_mesh_face_distance_unidirectional(
            gt_pointcloud,
            pred_mesh,
        )

    pred_to_gt_np = (
        pred_to_gt.detach().cpu().numpy().astype(np.float64)
    )
    gt_to_pred_np = (
        gt_to_pred.detach().cpu().numpy().astype(np.float64)
    )

    pred_mean = float(np.mean(pred_to_gt_np))
    gt_mean = float(np.mean(gt_to_pred_np))

    assd = 0.5 * (pred_mean + gt_mean)

    hd = max(
        float(np.max(pred_to_gt_np)),
        float(np.max(gt_to_pred_np)),
    )

    hd90 = max(
        float(np.percentile(pred_to_gt_np, 90)),
        float(np.percentile(gt_to_pred_np, 90)),
    )

    hd95 = max(
        float(np.percentile(pred_to_gt_np, 95)),
        float(np.percentile(gt_to_pred_np, 95)),
    )

    hd99 = max(
        float(np.percentile(pred_to_gt_np, 99)),
        float(np.percentile(gt_to_pred_np, 99)),
    )

    chamfer = pred_mean + gt_mean

    pred_squared_mean = float(
        np.mean(np.square(pred_to_gt_np))
    )
    gt_squared_mean = float(
        np.mean(np.square(gt_to_pred_np))
    )

    chamfer_squared = pred_squared_mean + gt_squared_mean

    surface_rmse = float(
        np.sqrt(
            0.5 * (
                pred_squared_mean +
                gt_squared_mean
            )
        )
    )

    return {
        "ASSD": float(assd),
        "HD": float(hd),
        "HD90": float(hd90),
        "HD95": float(hd95),
        "HD99": float(hd99),
        "Chamfer": float(chamfer),
        "ChamferSquared": float(chamfer_squared),
        "SurfaceRMSE": float(surface_rmse),
        "NumSurfaceSamples": int(n_points),
    }


def compute_surface_distance_metrics(
    pred_vertices: np.ndarray,
    pred_faces: np.ndarray,
    gt_vertices: np.ndarray,
    gt_faces: np.ndarray,
    n_points: int = 10000,
    device: Any = "cpu",
    seed: int = 42,
) -> Dict[str, float]:
    """
    Calculate symmetric sampled point-to-surface metrics.

    The target in each direction is the complete triangle mesh, rather than
    another sampled point cloud.
    """
    pred_vertices, pred_faces = _validate_mesh_arrays(
        pred_vertices,
        pred_faces,
    )
    gt_vertices, gt_faces = _validate_mesh_arrays(
        gt_vertices,
        gt_faces,
    )

    if not isinstance(device, torch.device):
        device = torch.device(device)

    return _compute_surface_distance_metrics_torch(
        pred_vertices=pred_vertices,
        pred_faces=pred_faces,
        gt_vertices=gt_vertices,
        gt_faces=gt_faces,
        n_points=n_points,
        device=device,
        seed=seed,
    )


def _canonical_face_set(faces: np.ndarray) -> np.ndarray:
    """
    Return faces in a representation independent of face winding and row
    ordering.
    """
    canonical = np.sort(np.asarray(faces, dtype=np.int64), axis=1)

    order = np.lexsort(
        (
            canonical[:, 2],
            canonical[:, 1],
            canonical[:, 0],
        )
    )

    return canonical[order]


def compute_correspondence_rmsd(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    pred_faces: Optional[np.ndarray] = None,
    gt_faces: Optional[np.ndarray] = None,
    require_identical_faces: bool = False,
) -> Dict[str, Any]:
    """
    Calculate true vertex-correspondence RMSD:

        RMSD = sqrt(mean(||v_pred_i - v_gt_i||^2))

    Vertex index i is assumed to refer to the same anatomical vertex in both
    meshes.

    If require_identical_faces is enabled, prediction and GT must additionally
    contain the same triangle set, ignoring triangle winding and row order.
    """
    pred_vertices = np.asarray(pred_vertices, dtype=np.float64)
    gt_vertices = np.asarray(gt_vertices, dtype=np.float64)

    if pred_vertices.ndim != 2 or pred_vertices.shape[1] != 3:
        return {
            "RMSD": np.nan,
            "RMSDValid": False,
            "RMSDError": "Prediction vertices do not have shape (V, 3).",
        }

    if gt_vertices.ndim != 2 or gt_vertices.shape[1] != 3:
        return {
            "RMSD": np.nan,
            "RMSDValid": False,
            "RMSDError": "GT vertices do not have shape (V, 3).",
        }

    if pred_vertices.shape[0] != gt_vertices.shape[0]:
        return {
            "RMSD": np.nan,
            "RMSDValid": False,
            "RMSDError": (
                "Prediction and GT vertex counts differ: {} versus {}."
            ).format(
                pred_vertices.shape[0],
                gt_vertices.shape[0],
            ),
        }

    if require_identical_faces:
        if pred_faces is None or gt_faces is None:
            return {
                "RMSD": np.nan,
                "RMSDValid": False,
                "RMSDError": (
                    "Identical faces were required, but face arrays "
                    "were not supplied."
                ),
            }

        pred_faces = np.asarray(pred_faces, dtype=np.int64)
        gt_faces = np.asarray(gt_faces, dtype=np.int64)

        if pred_faces.shape != gt_faces.shape:
            return {
                "RMSD": np.nan,
                "RMSDValid": False,
                "RMSDError": (
                    "Prediction and GT face-array shapes differ."
                ),
            }

        pred_canonical = _canonical_face_set(pred_faces)
        gt_canonical = _canonical_face_set(gt_faces)

        if not np.array_equal(pred_canonical, gt_canonical):
            return {
                "RMSD": np.nan,
                "RMSDValid": False,
                "RMSDError": (
                    "Prediction and GT do not have identical "
                    "triangle connectivity."
                ),
            }

    differences = pred_vertices - gt_vertices
    squared_euclidean = np.sum(
        np.square(differences),
        axis=1,
    )

    rmsd = float(np.sqrt(np.mean(squared_euclidean)))

    return {
        "RMSD": rmsd,
        "RMSDValid": True,
        "RMSDError": "",
    }


def compute_binary_overlap(
    prediction_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate Dice and IoU for two binary masks.

    If both masks are empty, Dice and IoU are defined as 1.
    """
    prediction_mask = np.asarray(
        prediction_mask,
        dtype=bool,
    )

    ground_truth_mask = np.asarray(
        ground_truth_mask,
        dtype=bool,
    )

    if prediction_mask.shape != ground_truth_mask.shape:
        raise ValueError(
            "Prediction and GT masks have different shapes: {} and {}."
            .format(
                prediction_mask.shape,
                ground_truth_mask.shape,
            )
        )

    intersection = int(
        np.count_nonzero(
            prediction_mask & ground_truth_mask
        )
    )

    prediction_count = int(
        np.count_nonzero(prediction_mask)
    )

    ground_truth_count = int(
        np.count_nonzero(ground_truth_mask)
    )

    union = int(
        np.count_nonzero(
            prediction_mask | ground_truth_mask
        )
    )

    denominator = prediction_count + ground_truth_count

    if denominator == 0:
        dice_score = 1.0
    else:
        dice_score = (
            2.0 * intersection / float(denominator)
        )

    if union == 0:
        iou_score = 1.0
    else:
        iou_score = intersection / float(union)

    return {
        "Dice": float(dice_score),
        "IoU": float(iou_score),
        "OverlapIntersectionVoxelCount": intersection,
        "OverlapUnionVoxelCount": union,
        "PredForegroundVoxelCount": prediction_count,
        "GTForegroundVoxelCount": ground_truth_count,
    }


# -------------------------------------------------------------------------
# Legacy project interfaces
# -------------------------------------------------------------------------


class EvalMetric(ABC):
    """
    Abstract compatibility base class used by the original project.
    """

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(
        self,
        mesh_pred,
        mesh_gt,
        n_m_classes,
        mesh_label_names,
        voxel_pred,
        voxel_gt,
        n_v_classes,
        voxel_label_names,
    ):
        pass

    def result_blueprint(self):
        return {
            "Tissue": [],
            "Metric": [],
            "Value": [],
        }


class SurfaceDistance(EvalMetric):
    """
    Compatibility wrapper around the new surface-distance implementation.
    """

    def __init__(self):
        self.n_points = 100000
        self.seed = 42

    def __call__(
        self,
        mesh_pred,
        mesh_gt,
        n_m_classes,
        mesh_label_names,
        voxel_pred,
        voxel_gt,
        n_v_classes,
        voxel_label_names,
    ):
        pred_vertices = mesh_pred.verts_list()
        pred_faces = mesh_pred.faces_list()

        gt_vertices = mesh_gt.verts_list()
        gt_faces = mesh_gt.faces_list()

        results = self.result_blueprint()

        for index, (
            name,
            pred_v,
            pred_f,
            gt_v,
            gt_f,
        ) in enumerate(
            zip(
                mesh_label_names,
                pred_vertices,
                pred_faces,
                gt_vertices,
                gt_faces,
            )
        ):
            metrics = _compute_surface_distance_metrics_torch(
                pred_vertices=pred_v,
                pred_faces=pred_f,
                gt_vertices=gt_v,
                gt_faces=gt_f,
                n_points=self.n_points,
                device=pred_v.device,
                seed=self.seed + index,
            )

            for metric_name in SURFACE_DISTANCE_METRIC_KEYS:
                results["Tissue"].append(name)
                results["Metric"].append(metric_name)
                results["Value"].append(
                    metrics[metric_name]
                )

        return results


class SelfIntersections(EvalMetric):
    """
    Compatibility wrapper for percentage of self-intersecting faces.
    """

    def __call__(
        self,
        mesh_pred,
        mesh_gt,
        n_m_classes,
        mesh_label_names,
        voxel_pred,
        voxel_gt,
        n_v_classes,
        voxel_label_names,
    ):
        pred_vertices = mesh_pred.verts_list()
        pred_faces = mesh_pred.faces_list()

        results = self.result_blueprint()

        for name, pred_v, pred_f in zip(
            mesh_label_names,
            pred_vertices,
            pred_faces,
        ):
            metrics = compute_self_intersections(
                pred_v.detach().cpu().numpy(),
                pred_f.detach().cpu().numpy(),
            )

            results["Tissue"].append(name)
            results["Metric"].append(
                "SelfIntersectingFacesPct"
            )
            results["Value"].append(
                metrics["SelfIntersectingFacesPct"]
            )

        return results


class VoxelDice(EvalMetric):
    """
    Compatibility implementation of label-wise voxel Dice.
    """

    def __call__(
        self,
        mesh_pred,
        mesh_gt,
        n_m_classes,
        mesh_label_names,
        voxel_pred,
        voxel_gt,
        n_v_classes,
        voxel_label_names,
    ):
        results = self.result_blueprint()

        for label_id, name in zip(
            range(1, n_v_classes),
            voxel_label_names,
        ):
            pred_mask = (
                voxel_pred == label_id
            ).detach().cpu().numpy()

            gt_mask = (
                voxel_gt == label_id
            ).detach().cpu().numpy()

            score = compute_binary_overlap(
                pred_mask,
                gt_mask,
            )["Dice"]

            results["Tissue"].append(name)
            results["Metric"].append("VoxelDice")
            results["Value"].append(score)

        return results


class MeshDice(EvalMetric):
    """
    Retained compatibility implementation of mesh-to-voxel Dice.
    """

    def __call__(
        self,
        mesh_pred,
        mesh_gt,
        n_m_classes,
        mesh_label_names,
        voxel_pred,
        voxel_gt,
        n_v_classes,
        voxel_label_names,
    ):
        # Imported lazily because the new surface-only evaluator does not
        # require the repository's voxelization dependencies.
        from utils.utils import voxelize_mesh

        pred_vertices = mesh_pred.verts_list()
        pred_faces = mesh_pred.faces_list()

        results = self.result_blueprint()

        for label_id, name in zip(
            range(1, n_v_classes),
            voxel_label_names,
        ):
            matching_names = [
                mesh_name
                for mesh_name in mesh_label_names
                if name in mesh_name
            ]

            matching_ids = [
                mesh_label_names.index(mesh_name)
                for mesh_name in matching_names
            ]

            pred_mesh = Meshes(
                verts=[
                    pred_vertices[index]
                    for index in matching_ids
                ],
                faces=[
                    pred_faces[index]
                    for index in matching_ids
                ],
            )

            pred_voxelized = voxelize_mesh(
                pred_mesh.verts_packed().cpu().numpy(),
                pred_mesh.faces_packed().cpu().numpy(),
                voxel_gt.shape,
            )

            pred_mask = (
                pred_voxelized == 1
            ).detach().cpu().numpy()

            gt_mask = (
                voxel_gt == label_id
            ).detach().cpu().numpy()

            score = compute_binary_overlap(
                pred_mask,
                gt_mask,
            )["Dice"]

            results["Tissue"].append(name)
            results["Metric"].append("MeshDice")
            results["Value"].append(score)

        return results


def Jaccard(pred, target, n_classes):
    """
    Legacy multiclass Jaccard/IoU implementation, ignoring background 0.
    """
    ious = []

    for class_id in range(1, n_classes):
        pred_mask = (
            pred == class_id
        ).detach().cpu().numpy()

        target_mask = (
            target == class_id
        ).detach().cpu().numpy()

        iou = compute_binary_overlap(
            pred_mask,
            target_mask,
        )["IoU"]

        ious.append(iou)

    if len(ious) == 0:
        return np.nan

    return float(np.mean(ious))