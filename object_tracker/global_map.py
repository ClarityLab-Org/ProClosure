import numpy as np

from .voxel_utils import compute_voxel, voxelize_points


def build_global_voxel_map(keyframes, voxel_size=0.02):
    """
    Build a sparse voxel map over all keyframes.

    Returned for backward compatibility and for visualization / export.
    The online tracker does NOT require this to run.
    """
    voxel_map = {}
    global_points = []

    for kf in keyframes:
        pts = kf.points_world.reshape(-1, 3)
        finite = np.isfinite(pts).all(axis=1)
        nonzero = np.linalg.norm(pts, axis=1) > 1e-6
        pts = pts[finite & nonzero]

        for pt in pts:
            voxel = compute_voxel(pt, voxel_size)
            if voxel not in voxel_map:
                idx = len(global_points)
                voxel_map[voxel] = idx
                global_points.append(pt)

    global_points = np.array(global_points) if global_points else np.zeros((0, 3))
    print(f"[INFO] Global voxel map built with {len(global_points)} voxels")
    return voxel_map, global_points


def object_points_to_voxels(points, voxel_size):
    """
    Convert an object's 3D points into the set of voxel tuples they occupy.
    This is the preferred API for the new tracker.
    """
    return voxelize_points(points, voxel_size)


def object_points_to_indices(points, voxel_map, voxel_size):
    """
    Backward-compatible helper: return integer indices into ``global_points``.
    Kept so existing pipeline code keeps running; new code should use
    ``object_points_to_voxels`` instead.
    """
    obj_voxels = voxelize_points(points, voxel_size)
    indices = set()
    for v in obj_voxels:
        if v in voxel_map:
            indices.add(voxel_map[v])
    return indices
