import numpy as np


def compute_voxel(p, voxel_size):
    """Convert a 3D point to its integer voxel coordinate."""
    return (
        int(np.floor(p[0] / voxel_size)),
        int(np.floor(p[1] / voxel_size)),
        int(np.floor(p[2] / voxel_size)),
    )


def voxelize_points(points, voxel_size):
    """Convert a dense point cloud into a set of voxel-coordinate tuples."""
    voxels = set()
    for p in points:
        voxels.add(compute_voxel(p, voxel_size))
    return voxels


def compute_ios(voxels_a, voxels_b):
    """
    Intersection-over-Smaller.

    Unlike IoU, the denominator is bounded by the smaller voxel set, so the
    score stays meaningful when one side (typically an accumulated track) has
    grown much larger than the other (a fresh observation of a partial view).
    """
    if not voxels_a or not voxels_b:
        return 0.0
    inter = len(voxels_a & voxels_b)
    denom = min(len(voxels_a), len(voxels_b))
    return inter / denom if denom > 0 else 0.0


def compute_iou(voxels_a, voxels_b):
    """Classic set IoU, kept for reporting / ablations."""
    if not voxels_a or not voxels_b:
        return 0.0
    inter = voxels_a & voxels_b
    union = voxels_a | voxels_b
    return len(inter) / len(union) if union else 0.0


def voxels_touching(voxels_a, voxels_b, k=2):
    """
    True if any voxel in A lies within Chebyshev distance <= k of any voxel in B.
    Early-exits on the first match. Iterates the smaller set for speed.
    """
    if not voxels_a or not voxels_b:
        return False
    if len(voxels_a) > len(voxels_b):
        voxels_a, voxels_b = voxels_b, voxels_a
    for (x, y, z) in voxels_a:
        for dx in range(-k, k + 1):
            for dy in range(-k, k + 1):
                for dz in range(-k, k + 1):
                    if (x + dx, y + dy, z + dz) in voxels_b:
                        return True
    return False


def dilated_neighbors(voxel, k):
    """Yield every voxel tuple within Chebyshev distance k (including center)."""
    x, y, z = voxel
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            for dz in range(-k, k + 1):
                yield (x + dx, y + dy, z + dz)
