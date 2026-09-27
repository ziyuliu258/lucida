from __future__ import annotations

import numpy as np

from .pose import Pose


def transform_normalized_points(points: np.ndarray, pose: Pose) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return (pose.rotation @ (points * pose.size).T).T + pose.position


def rotation_geodesic_deg(prediction: Pose, target: Pose) -> float:
    relative = prediction.rotation.T @ target.rotation
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def add_sb(predicted_points: np.ndarray, target_points: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    predicted_points = np.asarray(predicted_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    pred_to_target = cKDTree(target_points).query(predicted_points, workers=-1)[0].mean()
    target_to_pred = cKDTree(predicted_points).query(target_points, workers=-1)[0].mean()
    return float((pred_to_target + target_to_pred) / 2.0)


def object_diameter(points: np.ndarray) -> float:
    from scipy.spatial.distance import pdist

    points = np.asarray(points, dtype=np.float64)
    return float(pdist(points).max())


def oriented_box_corners(pose: Pose) -> np.ndarray:
    """Return the eight corners of the pose's oriented box in world coordinates."""
    signs = np.array(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        dtype=np.float64,
    )
    return (pose.rotation @ (signs * pose.size).T).T + pose.position


def oriented_box_iou(first: Pose, second: Pose, tolerance: float = 1e-9) -> float:
    """Compute 3D IoU of two oriented boxes from their convex intersection.

    The intersection is described by the 12 half-spaces of the two boxes.  A
    vertex of that convex polyhedron is an intersection of three boundary
    planes, so enumerate those combinations and retain the feasible ones.

    Working in coordinates centred between the boxes is deliberate: CA-1M
    poses use large world coordinates while some reconstructed meshes are only
    about 11 mm thick.  The previous edge/face method lost valid vertices in
    this near-coplanar case and reported a zero-volume intersection.
    """
    from scipy.spatial import ConvexHull, QhullError

    center = (first.position + second.position) / 2.0
    normals: list[np.ndarray] = []
    offsets: list[float] = []
    for box in (first, second):
        relative_position = box.position - center
        for axis in range(3):
            normal = box.rotation[:, axis]
            half_extent = float(box.size[axis] / 2.0)
            offset = float(normal @ relative_position)
            normals.extend((normal, -normal))
            offsets.extend((offset + half_extent, -offset + half_extent))

    plane_normals = np.asarray(normals, dtype=np.float64)
    plane_offsets = np.asarray(offsets, dtype=np.float64)
    candidates: list[np.ndarray] = []
    for first_plane in range(len(plane_normals)):
        for second_plane in range(first_plane + 1, len(plane_normals)):
            for third_plane in range(second_plane + 1, len(plane_normals)):
                indices = (first_plane, second_plane, third_plane)
                matrix = plane_normals[list(indices)]
                # Parallel or almost-parallel face triples have no unique vertex.
                if abs(float(np.linalg.det(matrix))) <= 1e-10:
                    continue
                point = np.linalg.solve(matrix, plane_offsets[list(indices)])
                if np.all(plane_normals @ point <= plane_offsets + tolerance):
                    candidates.append(point)

    if len(candidates) < 4:
        intersection = 0.0
    else:
        unique = np.unique(np.round(np.asarray(candidates), decimals=12), axis=0)
        if len(unique) < 4:
            intersection = 0.0
        else:
            try:
                intersection = float(ConvexHull(unique).volume)
            except QhullError:
                intersection = 0.0

    volume_a = float(np.prod(first.size))
    volume_b = float(np.prod(second.size))
    union = volume_a + volume_b - intersection
    return float(np.clip(intersection / union, 0.0, 1.0)) if union > 0.0 else 0.0


def evaluate_pose(
    normalized_surface_points: np.ndarray,
    prediction: Pose,
    target: Pose,
) -> dict[str, float]:
    predicted_points = transform_normalized_points(normalized_surface_points, prediction)
    target_points = transform_normalized_points(normalized_surface_points, target)
    distance = add_sb(predicted_points, target_points)
    diameter = object_diameter(target_points)
    fraction = distance / diameter if diameter > 0.0 else float("inf")
    rotation_error = rotation_geodesic_deg(prediction, target)
    return {
        "add_sb_m": distance,
        "object_diameter_m": diameter,
        "add_sb_fraction": fraction,
        "add_sb_at_0.10": float(fraction < 0.10),
        "add_sb_at_0.05": float(fraction < 0.05),
        "add_sb_at_0.01": float(fraction < 0.01),
        "iou_3d": oriented_box_iou(prediction, target),
        "rotation_error_deg": rotation_error,
        "rotation_at_5deg": float(rotation_error < 5.0),
    }
