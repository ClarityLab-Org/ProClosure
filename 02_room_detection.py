import os
import json
import yaml
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, asdict
from scipy.spatial.transform import Rotation as R_sci
from scipy.ndimage import binary_dilation
from scipy.ndimage import distance_transform_edt, label as cc_label, find_objects
from matplotlib.patches import Patch


# ══════════════════════════════════════════════════════════════════════════════
# BEV CALIBRATION — world ↔ pixel mapping
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class BEVCalibration:
    """Stores everything needed to convert 3D world points → BEV pixels."""
    x_min: float;  x_max: float      # world X bounds (col axis)
    z_min: float;  z_max: float      # world Z bounds (row axis)
    resolution: int                   # square image side length
    R_total: np.ndarray               # 3×3 alignment rotation
    center: np.ndarray                # 3D rotation center
    up_axis_idx: int = 1              # Y is up after alignment

    def to_dict(self):
        """Serialisable dict (for JSON save)."""
        return {
            "x_min": self.x_min, "x_max": self.x_max,
            "z_min": self.z_min, "z_max": self.z_max,
            "resolution": self.resolution,
            "R_total": self.R_total.tolist(),
            "center": self.center.tolist(),
            "up_axis_idx": self.up_axis_idx,
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstruct from saved JSON dict."""
        return cls(
            x_min=d["x_min"], x_max=d["x_max"],
            z_min=d["z_min"], z_max=d["z_max"],
            resolution=d["resolution"],
            R_total=np.array(d["R_total"]),
            center=np.array(d["center"]),
            up_axis_idx=d.get("up_axis_idx", 1),
        )

    # ---------- coordinate transforms ----------

    def world3d_to_aligned(self, pts_3d):
        """Apply R_total rotation about center. pts_3d: (N,3) or (3,)."""
        pts = np.atleast_2d(pts_3d)
        aligned = (self.R_total @ (pts - self.center).T).T + self.center
        return aligned.squeeze()

    def aligned_to_pixel(self, x, z):
        """Aligned world X,Z → pixel (row, col). Scalars or arrays."""
        col = (x - self.x_min) / (self.x_max - self.x_min) * (self.resolution - 1)
        row = (z - self.z_min) / (self.z_max - self.z_min) * (self.resolution - 1)
        col = np.clip(col, 0, self.resolution - 1).astype(int)
        row = np.clip(row, 0, self.resolution - 1).astype(int)
        return row, col

    def pixel_to_aligned(self, row, col):
        """Pixel (row, col) → aligned world (X, Z)."""
        x = self.x_min + col / (self.resolution - 1) * (self.x_max - self.x_min)
        z = self.z_min + row / (self.resolution - 1) * (self.z_max - self.z_min)
        return x, z

    def world3d_to_pixel(self, pts_3d):
        """Full chain: raw 3D centroid → pixel (row, col)."""
        aligned = self.world3d_to_aligned(pts_3d)
        aligned = np.atleast_2d(aligned)
        return self.aligned_to_pixel(aligned[:, 0], aligned[:, 2])


# ══════════════════════════════════════════════════════════════════════════════
# ─── PHASE A ──────────────────────────────────────────────────────────────────
# Full PCD alignment → Wall PCD → BEV image + calibration
# ══════════════════════════════════════════════════════════════════════════════

# A.1 — load & preprocess
def phase_a_load(ply_path, voxel_size, outlier_nb=20, outlier_std=2.0):
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"[A.1] raw points: {len(pcd.points)}")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=outlier_nb,
                                            std_ratio=outlier_std)
    print(f"[A.1] after preprocessing: {len(pcd.points)}")
    return pcd


# A.2 — gravity alignment (iterative RANSAC floor detection)
def gravity_aligned_pts(pcd, pcd_center, up_axis=np.array([0.0, 1.0, 0.0]),
                        verticality_thresh=0.85):
    """Find floor via iterative RANSAC, rotate so floor normal → up_axis."""
    work = o3d.geometry.PointCloud(pcd)
    if len(work.points) < 1000:
        raise RuntimeError(f"Point cloud too small ({len(work.points)} pts).")

    candidates = []
    remaining = work
    for i in range(6):
        if len(remaining.points) < 500:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=0.03, ransac_n=3, num_iterations=2000)
        a, b, c, _ = plane_model
        n = np.array([a, b, c]); n /= np.linalg.norm(n)
        print(f"[A.2] plane {i}: normal={n}, inliers={len(inliers)}")
        candidates.append((len(inliers), n))
        remaining = remaining.select_by_index(inliers, invert=True)

    if not candidates:
        raise RuntimeError("No planes detected.")

    horiz = [(cnt, n) for (cnt, n) in candidates
             if np.max(np.abs(n)) > verticality_thresh]
    if not horiz:
        horiz = sorted(candidates, key=lambda x: np.max(np.abs(x[1])),
                       reverse=True)[:1]
        print("[A.2] WARNING: weak axis-aligned plane.")

    horiz.sort(key=lambda x: x[0], reverse=True)
    _, floor_normal = horiz[0]
    print(f"[A.2] selected floor normal: {floor_normal}")

    if np.dot(floor_normal, up_axis) < 0:
        floor_normal = -floor_normal

    rot, _ = R_sci.align_vectors([up_axis], [floor_normal])
    R_mat = rot.as_matrix()
    pcd.rotate(R_mat, center=pcd_center)
    return pcd, R_mat


# A.3 — Manhattan correction (yaw)
def manhattan_align(pcd, pcd_center, up_axis_idx=1, wall_band=(0.25, 0.75),
                    n_bins=360, normal_radius=0.15):
    """Snap dominant wall direction via horizontal-normal histogram."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)

    u = pts[:, up_axis_idx]
    lo, hi = np.quantile(u, wall_band[0]), np.quantile(u, wall_band[1])
    band_mask = (u > lo) & (u < hi)

    horizontality = 1.0 - np.abs(nrm[:, up_axis_idx])
    wall_mask = band_mask & (horizontality > 0.85)
    wall_normals = nrm[wall_mask]
    print(f"[A.3] slab pts={band_mask.sum()}, wall normals={wall_mask.sum()}")

    if len(wall_normals) < 200:
        print("[A.3] too few wall normals, skipping")
        return pcd, np.eye(3), None

    horiz_axes = [i for i in range(3) if i != up_axis_idx]
    n2 = wall_normals[:, horiz_axes].copy()
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-9

    angles = np.arctan2(n2[:, 1], n2[:, 0])
    folded = np.mod(angles, np.pi / 2.0)
    hist, edges = np.histogram(folded, bins=n_bins, range=(0, np.pi / 2.0))
    peak_idx = int(np.argmax(hist))
    peak = edges[peak_idx] + (edges[1] - edges[0]) / 2.0
    print(f"[A.3] dominant angle = {np.degrees(peak):.2f}\u00b0")

    theta = -peak
    cs, sn = np.cos(theta), np.sin(theta)
    R_yaw = np.eye(3)
    a0, a1 = horiz_axes
    R_yaw[a0, a0] =  cs; R_yaw[a0, a1] = -sn
    R_yaw[a1, a0] =  sn; R_yaw[a1, a1] =  cs

    pcd.rotate(R_yaw, center=pcd_center)
    debug = {"hist": hist, "edges": edges, "peak_deg": float(np.degrees(peak))}
    return pcd, R_yaw, debug


# A.6 — rasterise a MID-HEIGHT SLAB of the wall cloud to BEV

def bw_topdown_map(pts, resolution, padding=0.0):
    """Cut a horizontal plane at the house's mid-height (Y = up axis after
    gravity alignment) and keep only the bottom half (Y <= y_mid),
    then project that bottom half's X,Z to a binary image.
    Returns (img, x_min, x_max, z_min, z_max)."""
    if len(pts) == 0:
        return (
            np.zeros((resolution, resolution), dtype=np.uint8),
            0.0, 0.0, 0.0, 0.0
        )

    y = pts[:, 1]                                   # Y is up (gravity-aligned)
    y_mid = 0.5 * (float(y.min()) + float(y.max()))  # middle of the height
    print("y_mid: ", y_mid)
    bottom_half = (y >= y_mid)
    slab = pts[bottom_half]

    print(
        f"[A.6] mid-height y={y_mid:.3f}, keeping bottom half -> "
        f"{len(slab)}/{len(pts)} pts below mid-plane"
    )

    if len(slab) == 0:
        slab = pts                                  # fallback if band is empty

    x, z = slab[:, 0], slab[:, 2]

    mn = np.array([x.min(), z.min()])
    mx = np.array([x.max(), z.max()])

    span = (mx - mn).max() * (1.0 + padding)

    ctr = (mn + mx) / 2.0
    mn = ctr - span / 2.0

    pix = span / resolution

    cols = np.clip(
        ((x - mn[0]) / pix).astype(int),
        0,
        resolution - 1
    )

    rows = np.clip(
        ((z - mn[1]) / pix).astype(int),
        0,
        resolution - 1
    )

    img = np.zeros((resolution, resolution), dtype=np.uint8)

    img[rows, cols] = 255

    return (
        img,
        float(mn[0]),
        float(mn[0] + span),
        float(mn[1]),
        float(mn[1] + span)
    )


# A.7 — denoise BEV (remove sparse noise, keep thick walls)
def phase_a_denoise_bev(bev_uint8, morph_kernel=3, min_component_area=50):
    """
    Clean up noise in the binary BEV image.

    1. Morphological opening: erode removes isolated thin pixels,
       dilate restores wall thickness. Walls survive because they are
       thick continuous structures.
    2. Connected-component area filter: remove blobs smaller than
       min_component_area pixels (noise from bad reconstruction).
    """
    # Step 1: morphological opening -- SKIPPED when morph_kernel <= 1.
    # At 256 px a wall line is only 1-2 px wide, so a 3x3 erode deletes it
    # outright: measured on testing_video5 the opening cut wall pixels from
    # 5146 to 4368 and broke the structure into 7 connected components instead
    # of 4, erasing most of the top room's walls. The component-area filter
    # below already removes isolated speckle, and it does so without eroding
    # long thin structures, so the opening is redundant at this resolution.
    if morph_kernel and morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (morph_kernel, morph_kernel))
        cleaned = cv2.morphologyEx(bev_uint8, cv2.MORPH_OPEN, kernel)
    else:
        cleaned = bev_uint8.copy()

    # Step 2: remove small connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8)

    output = np.zeros_like(cleaned)
    for i in range(1, n_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            output[labels == i] = 255

    removed_px = int(bev_uint8.sum() / 255 - output.sum() / 255)
    print(f"[A.7] denoise: removed {removed_px} noise pixels, "
          f"kept {int(output.sum() / 255)} wall pixels")
    return output


# A — orchestrator
def phase_a_pipeline(wall_ply_path, full_ply_path, cfg):
    """
    Returns: bev_uint8, calib (BEVCalibration)

    The calib object contains R_total + center + bounds, so later you can do:
        row, col = calib.world3d_to_pixel(object_centroid_3d)
    to assign objects to room cells.
    """
    voxel_size = cfg["voxel_size"]
    resolution = cfg["resolution"]

    # A.1 — load full cloud for alignment
    full_pcd = phase_a_load(full_ply_path, voxel_size)
    center = full_pcd.get_center().copy()

    # A.2 — gravity alignment (rotates full_pcd in-place)
    _, R_grav = gravity_aligned_pts(full_pcd, center)

    # A.3 — Manhattan correction (rotates full_pcd in-place)
    _, R_yaw, manhattan_dbg = manhattan_align(
        full_pcd, center,
        wall_band=tuple(cfg["wall_band"]),
        normal_radius=cfg["normal_radius"],
    )
    # A.4 — extra yaw about the up axis, folded into R_total.
    # Manhattan alignment snaps the dominant wall direction to an axis, but it
    # cannot tell which of the four 90 deg orientations is the one you want to
    # look at, so the plan can come out rotated. Applying it HERE rather than to
    # the saved image is what keeps everything consistent: R_total is what
    # BEVCalibration uses, so the wall cloud, the camera poses, the object
    # centroids and the pixel<->world mapping all rotate together.
    #
    # This is a proper rotation, not a mirror. Negating a single horizontal axis
    # would flip the handedness of the floor plan, i.e. produce a mirror image of
    # the real house.
    yaw_deg = float(cfg.get("bev_yaw_deg", 0.0))
    R_total = R_yaw @ R_grav
    if yaw_deg:
        th = np.radians(yaw_deg)
        cs, sn = np.cos(th), np.sin(th)
        R_flip = np.array([[cs, 0.0, sn],
                           [0.0, 1.0, 0.0],
                           [-sn, 0.0, cs]])          # rotation about Y (up)
        R_total = R_flip @ R_total
        print(f"[A.4] extra yaw {yaw_deg:.0f} deg about the up axis")

    # A.5 — apply same transform to wall cloud
    wall_pcd = o3d.io.read_point_cloud(wall_ply_path)
    print(f"[A.5] wall points: {len(wall_pcd.points)}")
    wall_pcd.rotate(R_total, center=center)
    wall_pts = np.asarray(wall_pcd.points)

    # A.6 — rasterise full wall to BEV (no slab cuts)
    bev_uint8, x_min, x_max, z_min, z_max = bw_topdown_map(
        wall_pts, resolution, padding=0.05)

    # A.7 — denoise: remove sparse noise pixels, keep thick wall lines
    bev_uint8 = phase_a_denoise_bev(
        bev_uint8,
        morph_kernel=cfg.get("morph_kernel", 3),
        min_component_area=cfg.get("min_component_area", 50),
    )

    # A.8 — build calibration
    calib = BEVCalibration(
        x_min=x_min, x_max=x_max,
        z_min=z_min, z_max=z_max,
        resolution=resolution,
        R_total=R_total,
        center=center,
    )

    debug = {
        "manhattan": manhattan_dbg,
        "n_wall_points": len(wall_pts),
        "n_full_points": len(full_pcd.points),
    }
    return bev_uint8, calib, debug

# ══════════════════════════════════════════════════════════════════════════════
# ─── PHASE B — camera-pose seeded room segmentation ──────────────────────────
#
# Replaces the Jul-21 Phase B. Validated on 10 HM3D floors against HOV-SG:
#
#   seeding    the seeds ARE the SLAM keyframe camera centres, not random or FPS
#              samples of the free space. A camera pose is free space by
#              construction -- over 10 HM3D floors, 3378 of 3378 on-floor poses
#              landed on a free pixel, none on a wall -- so rejection sampling,
#              the cardinal-enclosure tests, the candidate pool and the rng all
#              disappear and the method becomes DETERMINISTIC.
#              mean IoU 0.694 vs 0.668 for FPS seeding vs 0.631 for HOV-SG.
#
#   freezing   a pocket seals when SATURATED -- when it holds no pixel a new
#              seed could validly occupy -- instead of when its free component
#              stops touching the image border. Saturation fires while the seeds
#              are still spread at their clearance limit, so the sealed pocket is
#              full size rather than eaten away by dilation.
#
# Dilation, relocation, clustering and ray pruning are unchanged, and are
# imported from room_detection_camera_pose.py rather than duplicated so the two
# cannot drift. Repoint ROOM_MODULE_DIR if this file moves.
# ══════════════════════════════════════════════════════════════════════════════
import sys
from collections import deque

ROOM_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOM_MODULE_DIR not in sys.path:
    sys.path.insert(0, ROOM_MODULE_DIR)
try:
    import room_detection_camera_pose as rdp
except ImportError as e:                                    # pragma: no cover
    raise ImportError(
        "cannot import room_detection_camera_pose from {}. Set ROOM_MODULE_DIR "
        "to the directory holding it.".format(ROOM_MODULE_DIR)) from e


def load_camera_poses(path):
    """TUM-format trajectory -> (N, 3) camera centres in the SLAM world frame.

    Each row is `timestamp tx ty tz qx qy qz qw`, so the translation is columns
    1-3. These are the frame the fused cloud is in, so Phase A's calibration
    maps them to pixels with no extra registration -- verified on
    testing_video5, where every pose translation falls inside the cloud's bbox.
    """
    P = np.loadtxt(path)
    P = np.atleast_2d(P)
    if P.shape[1] < 4:
        raise ValueError(
            "{} has {} columns; expected TUM format "
            "`timestamp tx ty tz qx qy qz qw`".format(path, P.shape[1]))
    return P[:, 1:4].astype(np.float64)


def phase_b_footprint(full_ply_path, calib, cfg_a, blur=21, close_k=5, iters=3):
    """Filled building footprint on the calibration grid.

    Phase A rasterises WALLS only, so everything outside the building is free
    space in that image. Without a footprint the room fill in STEP 11 flows out
    through any gap in the outer wall and wraps a room right around the
    building. This is HOV-SG's outside-boundary construction, run on the full
    cloud.

    The cloud is re-loaded and re-aligned with calib.R_total rather than being
    threaded out of Phase A, so Phase A stays exactly as it was.
    """
    pcd = o3d.io.read_point_cloud(full_ply_path)
    pcd = pcd.voxel_down_sample(voxel_size=cfg_a.get("voxel_size", 0.05))
    pts = calib.world3d_to_aligned(np.asarray(pcd.points))

    # the footprint wants the building outline, so it keeps almost the whole
    # height -- only the very top is dropped, using the same mid-plane
    # convention as bw_topdown_map
    y = pts[:, 1]
    y_mid = 0.5 * (float(y.min()) + float(y.max()))
    slab = pts[y >= y_mid]
    if len(slab) == 0:
        slab = pts

    res = calib.resolution
    rows, cols = calib.aligned_to_pixel(slab[:, 0], slab[:, 2])
    hist = np.zeros((res, res), np.float32)
    np.add.at(hist, (rows, cols), 1.0)
    u8 = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (blur, blur), 2)
    _, fp = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY)
    fp = cv2.morphologyEx(fp, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT,
                                                    (close_k, close_k)),
                          iterations=iters)
    contours, _ = cv2.findContours(fp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    fp = np.zeros_like(fp)
    cv2.drawContours(fp, contours, -1, 255, -1)
    print(f"[B.1] footprint: {int((fp > 0).sum())} px of {res * res} "
          f"({100.0 * (fp > 0).mean():.1f}%)")
    return fp


def seeds_from_poses(pose_px, wall0, min_sep, clearance, stride=1,
                     verbose=True):
    """Camera poses -> seed dicts. No rng, no candidate pool, no rejection loop.

    Two filters, and only two:
      clearance  the pose needs `clearance` px of free space around it, the same
                 quantity the dilation and saturation tests use.
      min_sep    consecutive keyframes are centimetres apart, so without a
                 spacing filter one room would hold hundreds of seeds and the
                 saturation test would never fire.
    """
    from scipy.ndimage import distance_transform_edt

    H, W = wall0.shape
    dt = distance_transform_edt(1 - wall0)
    seeds = []
    ms2 = min_sep * min_sep
    n_out = n_wall = n_tight = n_close = 0
    for r, c in pose_px[::stride]:
        r, c = int(r), int(c)
        if not (0 <= r < H and 0 <= c < W):
            n_out += 1
        elif wall0[r, c]:
            n_wall += 1
        elif dt[r, c] < clearance:
            n_tight += 1
        elif any((r - t["pos"][0]) ** 2 + (c - t["pos"][1]) ** 2 < ms2
                 for t in seeds):
            n_close += 1
        else:
            seeds.append({"id": len(seeds), "pos": [r, c], "init_pos": (r, c),
                          "frozen": False, "cluster": -1})
            continue
    if verbose:
        print(f"[B.2] {len(pose_px[::stride])} poses -> {len(seeds)} seeds "
              f"(dropped: {n_out} off-image, {n_wall} on a wall, {n_tight} "
              f"under {clearance} px clearance, {n_close} within {min_sep} px "
              f"of a kept seed)")
    return seeds


def phase_b_pipeline(bev_uint8, calib, cfg_a, cfg_b, camera_poses_path,
                     full_ply_path, out_dir=None, verbose=True):
    """Camera poses -> seeds -> rooms. Returns (ctx, rooms).

    ctx carries wall_mask / footprint / labels_img so Phase C assigns objects
    without recomputing anything.
    """
    pose_world = load_camera_poses(camera_poses_path)
    rows, cols = calib.world3d_to_pixel(pose_world)
    pose_px = np.stack([np.atleast_1d(rows), np.atleast_1d(cols)], axis=1)
    if verbose:
        print(f"[B.1] {len(pose_world)} camera poses loaded from "
              f"{os.path.basename(camera_poses_path)}")

    wall0 = (bev_uint8 > 127).astype(np.uint8)
    wall0 = rdp.denoise_walls(wall0,
                              max_noise_area=cfg_b.get("max_noise_area", 200),
                              isolation_px=cfg_b.get("isolation_px", 15),
                              verbose=False)
    footprint = phase_b_footprint(full_ply_path, calib, cfg_a)

    min_sep = cfg_b.get("min_sep", 5)
    clearance = cfg_b.get("clearance", 5)
    seeds = seeds_from_poses(pose_px, wall0, min_sep, clearance,
                             cfg_b.get("pose_stride", 1), verbose)
    if not seeds:
        raise RuntimeError(
            "no camera pose survived seeding. Check that the poses and the "
            "cloud share a world frame, and that clearance is not larger than "
            "the rooms are wide (clearance={} px).".format(clearance))

    # steps 4-10, unchanged -- push this run's settings onto the shared module
    for k, v in (("MIN_SEP", min_sep), ("CLEARANCE", clearance),
                 ("DILATE_PX", cfg_b.get("dilate_px", 1)),
                 ("MAX_ITERS", cfg_b.get("max_iters", 600)),
                 ("MOVE_CAP", cfg_b.get("move_cap", 400)),
                 ("OUTLIER_K", cfg_b.get("outlier_k", 2.0)),
                 ("OUTLIER_MIN_N", cfg_b.get("outlier_min_n", 4)),
                 ("N_RAYS", cfg_b.get("n_corners", 50)),
                 ("RAY_LEN", cfg_b.get("ray_len", 150))):
        setattr(rdp, k, v)

    seeds, wall, wall0, frozen, bbox = rdp.run(
        out=out_dir or ".", img=(wall0 * 255).astype(np.uint8),
        figures=False, verbose=verbose, seed_fn=lambda *_a: seeds)

    erode = cfg_b.get("interior_erode_px", 6)
    interior = footprint
    if erode:
        interior = cv2.erode(footprint, cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * erode + 1,) * 2))

    K = max((s["cluster"] for s in seeds), default=-1) + 1
    labels_img, n_rooms = rdp.dense_room_labels(
        seeds, wall0, frozen, K, interior=interior,
        mode=cfg_b.get("label_mode", "rays"),
        min_room_px=cfg_b.get("min_room_px", 40))
    if verbose:
        print(f"[B] {K} clusters -> {n_rooms} rooms")

    rooms = [[] for _ in range(K)]
    for s in seeds:
        s["r"], s["c"] = int(s["init_pos"][0]), int(s["init_pos"][1])
        rooms[s["cluster"]].append(s)
    rooms = [cl for cl in rooms if cl]

    ctx = {"H": wall0.shape[0], "W": wall0.shape[1], "wall_mask": wall0,
           "footprint": footprint, "interior": interior,
           "labels_img": labels_img, "n_rooms": n_rooms, "frozen": frozen,
           "pose_px": pose_px, "bbox": bbox}
    return ctx, rooms


# ══════════════════════════════════════════════════════════════════════════════
# ─── PHASE C — object to room by surface majority ────────────────────────────
#
# Replaces the ray-intersection vote. Over 2255 annotated HM3D instances,
# surface majority scored ARI 0.668 against 0.640 for visibility voting in its
# best form (rays clipped to the footprint), so the simpler rule is used.
#
# An object takes the room most of ITS OWN points fall in, not the room holding
# its centroid. Concave and wall-mounted objects -- doors, mirrors, cabinets
# against a wall -- routinely have a centroid inside a wall while their surface
# is unambiguously in one room.
# ══════════════════════════════════════════════════════════════════════════════
def nearest_room_map(labels_img, wall0):
    """Nearest room for every pixel, measured THROUGH free space where possible.

    A straight-line nearest room is wrong for exactly the objects that need it:
    something inside a wall can be closer to the room on the far side. Rooms are
    grown with an 8-connected BFS restricted to free space, which cannot cross a
    wall; pixels no free path reaches fall back to Euclidean.
    """
    from scipy.ndimage import distance_transform_edt

    H, W = labels_img.shape
    free = wall0 == 0
    out = np.zeros_like(labels_img)
    out[labels_img > 0] = labels_img[labels_img > 0]

    q = deque()
    ys, xs = np.where((out > 0) & free)
    for y, x in zip(ys, xs):
        q.append((int(y), int(x)))
    while q:
        r, c = q.popleft()
        v = out[r, c]
        for dr, dc, _w in rdp.NEIGH8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == 0 and free[nr, nc]:
                out[nr, nc] = v
                q.append((nr, nc))
    if (out == 0).any() and (out > 0).any():
        _d, (ir, ic) = distance_transform_edt(out == 0, return_indices=True)
        gap = out == 0
        out[gap] = out[ir[gap], ic[gap]]
    return out


def object_points_from_obb(obb, n=5):
    """Sample a grid inside an oriented bounding box -> (n^3, 3) world points.

    thumbnails_metadata.json stores each object as a centroid plus an OBB, not
    as points -- ObjectTrack.voxels is dropped when the file is written. A single
    centroid is a poor vote: concave and wall-mounted objects (doors, mirrors,
    a bed against a wall) routinely have a centroid inside a wall while their
    extent is unambiguously in one room. Sampling the box recovers that extent,
    which is what the majority vote needs.
    """
    c = np.asarray(obb["center"], float)
    e = np.asarray(obb["extent"], float)
    R = np.asarray(obb["R"], float)
    t = np.linspace(-0.5, 0.5, n)
    g = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)
    return (R @ (g * e).T).T + c


def objects_from_metadata(path, obb_samples=5):
    """thumbnails_metadata.json -> [{id, label, points}].

    Entry layout written by object_tracker.save_objects:
        [cls, owner, centroid_xyz, obb | None, bbox_2d, thumb_path]
    """
    with open(path) as f:
        raw = json.load(f)
    out = []
    n_obb = 0
    for gid, e in raw.items():
        label, centroid, obb = e[0], np.asarray(e[2], float), e[3]
        if obb is not None:
            pts = object_points_from_obb(obb, obb_samples)
            n_obb += 1
        else:
            pts = centroid.reshape(1, 3)
        if not np.isfinite(pts).all():
            continue
        out.append({"id": int(gid), "label": label, "points": pts})
    print(f"[C] {len(out)} objects from {os.path.basename(path)} "
          f"({n_obb} with an OBB, {len(out) - n_obb} centroid only)")
    return out


def objects_from_obj_map(obj_map, voxel_size):
    """Live ObjectMap -> [{id, label, points}], using each track's voxels.

    Preferred over the metadata file when available: the voxels are the object's
    actual occupied volume rather than a box approximation of it.
    """
    out = []
    for gid in sorted(obj_map.tracks):
        trk = obj_map.tracks[gid]
        if getattr(trk, "voxels", None):
            v = np.asarray(sorted(trk.voxels), float)
            pts = (v + 0.5) * voxel_size
        elif getattr(trk, "centroid_world", None) is not None:
            pts = np.asarray(trk.centroid_world, float).reshape(1, 3)
        else:
            continue
        out.append({"id": int(gid), "label": trk.cls, "points": pts})
    print(f"[C] {len(out)} objects from the live ObjectMap")
    return out


def phase_c_pipeline(objects_in, ctx, calib, verbose=True):
    """Assign each object to the room most of its own points fall in.

    objects_in is [{id, label, points}] from either objects_from_metadata() or
    objects_from_obj_map(), so the assignment rule is identical whichever source
    the objects came from.
    """
    labels_img = ctx["labels_img"]
    near_img = nearest_room_map(labels_img, ctx["wall_mask"])

    objects, n_fallback = [], 0
    for o in objects_in:
        pts_w = np.atleast_2d(o["points"])
        rows, cols = calib.world3d_to_pixel(pts_w)
        rows, cols = np.atleast_1d(rows), np.atleast_1d(cols)

        lab = labels_img[rows, cols]
        lab = lab[lab > 0]
        if len(lab):
            vals, cnt = np.unique(lab, return_counts=True)
            room, method = int(vals[np.argmax(cnt)]), "majority"
            support = float(cnt.max()) / len(rows)
        else:
            vals, cnt = np.unique(near_img[rows, cols], return_counts=True)
            room, method, support = int(vals[np.argmax(cnt)]), "nearest", 0.0
            n_fallback += 1

        cen = pts_w.mean(axis=0)
        cr, cc = calib.world3d_to_pixel(np.atleast_2d(cen))
        objects.append({
            "id": o["id"], "label": o["label"], "room": room - 1,
            "method": method, "support": round(support, 3),
            "pixel": [int(np.atleast_1d(cr)[0]), int(np.atleast_1d(cc)[0])],
            "centroid_3d": cen.tolist(), "n_points": int(len(pts_w)),
        })
        if verbose:
            print(f"[C] {str(o['label']):>14s} (id={o['id']:3d}) -> room "
                  f"{room - 1} ({method}, support {support:.2f})")

    if verbose:
        print(f"[C] {len(objects)} objects assigned, {n_fallback} by the "
              f"nearest-room fallback")
        for ri in range(ctx["n_rooms"]):
            got = [o["label"] for o in objects if o["room"] == ri]
            print(f"[C] room {ri}: {got}")
    return objects


def phase_c_save_metadata(rooms, objects, ctx, calib, output_path):
    """room_metadata.json -- rooms, their seeds, and the object-room map."""
    rooms_summary = {}
    for ri, cluster in enumerate(rooms):
        rooms_summary[f"room_{ri}"] = {
            "n_seeds": len(cluster),
            "n_pixels": int((ctx["labels_img"] == ri + 1).sum()),
            "seeds": [{"r": s["r"], "c": s["c"]} for s in cluster],
        }
    metadata = {
        "R_total": calib.R_total.tolist(),
        "pcd_center": calib.center.tolist(),
        "bev_calibration": {
            "x_min": calib.x_min, "x_max": calib.x_max,
            "z_min": calib.z_min, "z_max": calib.z_max,
            "resolution": calib.resolution,
        },
        "n_rooms": ctx["n_rooms"],
        "rooms": rooms_summary,
        "object_room_map": {str(o["id"]): f"room_{o['room']}" for o in objects},
        "objects": {str(o["id"]): o for o in objects},
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[C] Saved room metadata -> {output_path}")
    return metadata


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
COLORS = [(80, 200, 110), (70, 150, 220), (210, 160, 50), (200, 80, 180),
          (60, 190, 200), (220, 90, 80), (150, 220, 80), (255, 160, 80),
          (100, 100, 220), (80, 220, 180), (180, 120, 220), (120, 200, 160),
          (220, 200, 90), (90, 160, 250), (250, 120, 160)]


def visualise(ctx, rooms, objects, save_path="./room_result.png",
              flip="lr"):
    """Trajectory + seeds | room segmentation | objects coloured by room."""
    H, W = ctx["H"], ctx["W"]
    labels_img, n_rooms = ctx["labels_img"], ctx["n_rooms"]
    pal = np.array(COLORS, float) / 255.0

    # DISPLAY-only reorientation. bw_topdown_map maps row = (z - z_min)/pix and
    # imshow draws row 0 at the top, and the sign of X and Z after the Manhattan
    # yaw is arbitrary, so the plan can come out in any of the eight dihedral
    # orientations. Which one is "right" is not derivable -- it is how the house
    # is meant to be viewed -- so it is a setting.
    #
    # Applied here and not to the raster or the cloud: BEVCalibration maps world
    # to pixel in the raster's frame, so reorienting either of those would
    # invalidate every pixel coordinate in room_metadata.json and break
    # pixel_to_aligned. 02_room_detection_old.py did the same thing with
    # np.flipud + _flip_r.
    _fx = flip in ("lr", "rot180")
    _fy = flip in ("ud", "rot180")

    def _pr(r):
        r = np.asarray(r, float)
        return (H - 1) - r if _fy else r

    def _pc(c):
        c = np.asarray(c, float)
        return (W - 1) - c if _fx else c

    def canvas(fill=False):
        img = np.ones((H, W, 3))
        if fill:
            for k in range(1, n_rooms + 1):
                img[labels_img == k] = 1.0 - 0.25 * (1.0 - pal[(k - 1) % len(pal)])
        img[ctx["wall_mask"] == 1] = (0.13, 0.13, 0.13)
        if _fx:
            img = np.fliplr(img)
        if _fy:
            img = np.flipud(img)
        return img

    if W > H * 1.4:
        fig, axes = plt.subplots(3, 1, figsize=(13, (13.0 * H / W + 1.2) * 3))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(21, 7.6))
    axes = np.atleast_1d(axes).ravel()

    pr = _pr(ctx["pose_px"][:, 0])
    pc = _pc(ctx["pose_px"][:, 1])
    axes[0].imshow(canvas(), interpolation="nearest")
    axes[0].plot(pc, pr, "-", color="#2b8cbe", lw=1.0, alpha=0.85, zorder=2)
    axes[0].scatter(pc, pr, s=5, color="#2b8cbe", alpha=0.5, linewidths=0,
                    zorder=3)
    for ri, cluster in enumerate(rooms):
        axes[0].scatter(_pc([s["c"] for s in cluster]),
                        _pr([s["r"] for s in cluster]),
                        s=58, color=pal[ri % len(pal)], edgecolor="white",
                        linewidths=1.0, zorder=4)
    n_seeds = sum(len(c) for c in rooms)
    on_wall = int(ctx["wall_mask"][
        np.clip(ctx["pose_px"][:, 0], 0, H - 1),
        np.clip(ctx["pose_px"][:, 1], 0, W - 1)].sum())
    axes[0].set_title(f"Camera trajectory and seeds\n{len(pr)} poses, "
                      f"{n_seeds} seeds, {on_wall} poses on a wall pixel",
                      fontsize=12)

    axes[1].imshow(canvas(fill=True), interpolation="nearest")
    axes[1].set_title(f"Room segmentation\n{n_rooms} rooms", fontsize=12)

    axes[2].imshow(canvas(fill=True), interpolation="nearest")
    for o in objects:
        r, c = float(_pr(o["pixel"][0])), float(_pc(o["pixel"][1]))
        axes[2].scatter([c], [r], s=72, color=pal[o["room"] % len(pal)],
                        edgecolor="white", linewidths=0.8, zorder=4,
                        marker="o" if o["method"] == "majority" else "^")
        axes[2].text(c + 5, r - 5, str(o["label"]), fontsize=6, color="black")
    n_fb = sum(1 for o in objects if o["method"] == "nearest")
    axes[2].set_title(f"Objects assigned to rooms\n{len(objects)} objects, "
                      f"{n_fb} by fallback (triangles)", fontsize=12)

    for ax in axes:
        ax.axis("off")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[viz] -> {save_path}  (display orientation: {flip})")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Phase A -> Phase B -> Phase C
# ══════════════════════════════════════════════════════════════════════════════
def run_all(config, obj_map=None, output_dir=None, verbose=True):
    """Phase A -> B -> C from a loaded config dict.

    obj_map is the live ObjectMap from 01_pipeline.py. Phase C needs the
    per-object voxels it carries, which thumbnails_metadata.json does not store,
    so pass it in rather than reading from disk. With obj_map=None the rooms are
    produced and Phase C is skipped.
    """
    cfg_a, cfg_b = config["phase_a"], config["phase_b"]
    output_dir = output_dir or config.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Phase A ──
    bev_uint8, calib, _dbg = phase_a_pipeline(
        os.path.expanduser(config["wall_ply_path"]),
        os.path.expanduser(config["full_ply_path"]), cfg_a)
    print(f"[A] BEV: {calib.resolution}x{calib.resolution} px")
    cv2.imwrite(os.path.join(output_dir, "bev_wall.png"), bev_uint8)
    # same image in the plan-view orientation the figures use, for the paper.
    # bev_wall.png itself stays in the calibration's frame.
    _fl = config.get("bev_display_flip", "lr")
    _v = bev_uint8
    if _fl in ("lr", "rot180"):
        _v = np.fliplr(_v)
    if _fl in ("ud", "rot180"):
        _v = np.flipud(_v)
    cv2.imwrite(os.path.join(output_dir, "bev_wall_planview.png"), _v)
    with open(os.path.join(output_dir, "phase_a_calib.json"), "w") as f:
        json.dump(calib.to_dict(), f, indent=2)

    # ── Phase B ──
    ctx, rooms = phase_b_pipeline(
        bev_uint8, calib, cfg_a, cfg_b,
        os.path.expanduser(config["camera_poses_path"]),
        os.path.expanduser(config["full_ply_path"]),
        out_dir=output_dir, verbose=verbose)

    # ── Phase C ──
    # the live ObjectMap is preferred (real voxels); the metadata file is the
    # standalone path and approximates each object by its OBB
    objects_in = None
    if obj_map is not None:
        objects_in = objects_from_obj_map(obj_map, cfg_a.get("voxel_size", 0.05))
    else:
        meta = config.get("thumbnails_metadata_path")
        meta = os.path.expanduser(meta) if meta else None
        if meta and os.path.exists(meta):
            objects_in = objects_from_metadata(meta)
        else:
            print(f"[C] skipped -- no ObjectMap passed and "
                  f"thumbnails_metadata_path not found ({meta})")

    objects = []
    if objects_in:
        objects = phase_c_pipeline(objects_in, ctx, calib, verbose)
        phase_c_save_metadata(rooms, objects, ctx, calib,
                              os.path.join(output_dir, "room_metadata.json"))

    visualise(ctx, rooms, objects,
              save_path=os.path.join(output_dir, "room_result.png"),
              flip=config.get("bev_display_flip", "lr"))
    print(f"[Done] {ctx['n_rooms']} rooms, {len(objects)} objects assigned.")
    return ctx, rooms, objects, calib


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "config", "config_thesis.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    run_all(config)
