"""
Online object tracker for monocular semantic object-centric SLAM.

Key ideas
---------
1. IoS (Intersection-over-Smaller) instead of IoU. As a track accumulates
   voxels across keyframes, IoU's denominator grows monotonically and the
   score degrades toward zero for genuine matches -- forcing an absurdly low
   threshold that then admits cross-object false merges. IoS keeps the
   denominator bounded by the smaller voxel set, so a single threshold (~0.3)
   stays meaningful for the full lifetime of a track.

2. Voxel -> tracks spatial index. Candidate tracks for a new observation are
   looked up in O(|obs.voxels| * (2k+1)^3) instead of O(|tracks|), which is
   required for real-time robot deployment.

3. Multi-match bridge merge. When a new observation spatially matches TWO
   existing tracks, those tracks are fused on the spot. This incrementally
   recovers the case where a track's observation chain was broken by a
   viewpoint gap (e.g. the front and back of a laptop were first seen with
   non-overlapping keyframes, and a later keyframe now spans both).

4. Bounded rolling merge. An optional second pass, run after each keyframe,
   pairwise-merges only the tracks that were actually updated in this step
   against other recent tracks. Cost is O(|updated| * |recent|), not O(N^2).
"""

import numpy as np
import open3d as o3d

from .voxel_utils import (
    compute_ios,
    compute_iou,
    voxels_touching,
    dilated_neighbors,
    voxelize_points,
)


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #
class ObjectObservation:
    """
    A single SAM3 detection back-projected into the global voxel grid.

    You can pass either:
      * ``voxels`` directly (a set / iterable of (vx, vy, vz) tuples), or
      * ``points_world`` + ``voxel_size`` and we voxelize for you.
    """

    def __init__(
        self,
        label,
        points_world,
        bbox_2d,
        img_h,
        img_w,
        score,
        num_points,
        mask_area_norm,
        voxels=None,
        voxel_size=None,
        thumbnail=None,
        frame_id=None,
    ):
        self.cls = label
        self.points_world = points_world

        if voxels is not None:
            self.voxels = set(voxels)
        elif voxel_size is not None and points_world is not None and len(points_world) > 0:
            self.voxels = voxelize_points(points_world, voxel_size)
        else:
            self.voxels = set()

        self.bbox_2d = bbox_2d
        self.img_h = img_h
        self.img_w = img_w

        self.score = score
        self.num_points = num_points
        self.mask_area_norm = mask_area_norm

        self.thumbnail = thumbnail
        self.frame_id = frame_id


# --------------------------------------------------------------------------- #
# Track
# --------------------------------------------------------------------------- #
class ObjectTrack:
    """An accumulated hypothesis for one physical object, spanning keyframes."""

    def __init__(self, track_id, obs):
        self.track_id = track_id
        self.cls = obs.cls

        # accumulated geometry (set of voxel tuples)
        self.voxels = set(obs.voxels)

        # best representative view
        self.best_num_points = obs.num_points
        self.best_mask_area = obs.mask_area_norm
        self.thumbnail = obs.thumbnail
        self.bbox_2d = obs.bbox_2d
        self.best_frame_id = obs.frame_id

        # per-frame bookkeeping
        self.visible_frames = {obs.frame_id}
        self.frame_bboxes = {obs.frame_id: obs.bbox_2d}
        self.last_seen_frame = obs.frame_id if obs.frame_id is not None else -1

        # live centroid for scene graph / NL queries
        self.centroid_world = None
        if obs.points_world is not None and len(obs.points_world) > 0:
            self.centroid_world = np.asarray(obs.points_world).mean(axis=0)

        self.obb_3d = None
        self.owner = None

    # ------------------------------------------------------------------ #
    # mutation
    # ------------------------------------------------------------------ #
    def absorb_observation(self, obs):
        """Merge a new observation into this track. Returns newly-added voxels."""
        new_voxels = obs.voxels - self.voxels
        self.voxels |= obs.voxels

        if obs.points_world is not None and len(obs.points_world) > 0:
            obs_c = np.asarray(obs.points_world).mean(axis=0)
            if self.centroid_world is None:
                self.centroid_world = obs_c
            else:
                n = len(self.visible_frames)
                self.centroid_world = (self.centroid_world * n + obs_c) / (n + 1)

        self.visible_frames.add(obs.frame_id)
        self.frame_bboxes[obs.frame_id] = obs.bbox_2d
        if obs.frame_id is not None:
            self.last_seen_frame = max(self.last_seen_frame, obs.frame_id)

        # upgrade representative view if this observation is better
        if (
            obs.mask_area_norm > self.best_mask_area * 1.1
            or obs.num_points > self.best_num_points * 1.1
        ):
            self.best_num_points = obs.num_points
            self.best_mask_area = obs.mask_area_norm
            self.thumbnail = obs.thumbnail
            self.bbox_2d = obs.bbox_2d
            self.best_frame_id = obs.frame_id

        return new_voxels

    def absorb_track(self, other):
        """Fuse another track into this one. Returns newly-added voxels."""
        new_voxels = other.voxels - self.voxels
        self.voxels |= other.voxels
        self.visible_frames |= other.visible_frames
        self.frame_bboxes.update(other.frame_bboxes)
        self.last_seen_frame = max(self.last_seen_frame, other.last_seen_frame)

        if other.best_mask_area > self.best_mask_area:
            self.best_num_points = other.best_num_points
            self.best_mask_area = other.best_mask_area
            self.thumbnail = other.thumbnail
            self.bbox_2d = other.bbox_2d
            self.best_frame_id = other.best_frame_id

        if other.centroid_world is not None:
            if self.centroid_world is None:
                self.centroid_world = other.centroid_world
            else:
                w_self = max(len(self.visible_frames), 1)
                w_other = max(len(other.visible_frames), 1)
                w = w_self + w_other
                self.centroid_world = (
                    self.centroid_world * w_self + other.centroid_world * w_other
                ) / w

        return new_voxels

    def set_owner(self, owner_name):
        self.owner = owner_name

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def compute_geometry(self, voxel_size):
        """Compute centroid + oriented bounding box from accumulated voxels."""
        if len(self.voxels) < 10:
            return
        pts = np.array(
            [(vx + 0.5, vy + 0.5, vz + 0.5) for (vx, vy, vz) in self.voxels],
            dtype=np.float64,
        ) * voxel_size

        self.centroid_world = pts.mean(axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        try:
            self.obb_3d = pcd.get_oriented_bounding_box()
        except Exception:
            self.obb_3d = None


# --------------------------------------------------------------------------- #
# Global map (online tracker)
# --------------------------------------------------------------------------- #
class GlobalObjectMap:
    """
    Online object tracker.

    Parameters
    ----------
    ios_thresh : float
        Single Intersection-over-Smaller threshold used for both:
          (a) matching a new observation to an existing track, and
          (b) fusing two existing tracks of the same class.
        Without spatial dilation (touch_k=0), IoS is only > 0 when two voxel
        sets share *identical* voxel coordinates. Two distinct same-class
        objects sitting next to each other (e.g. two remotes on a table)
        therefore have IoS == 0 and will never match. A small value
        (~0.05) is enough to accept partial re-observations of the same
        physical object.
    touch_k : int
        Chebyshev radius used for the spatial-index candidate lookup.
        Default is 0 (no dilation): a track is considered a candidate only
        if it shares at least one *actual* voxel with the new observation.
        Set to >0 only if SLAM pose drift causes the same surface to land
        in different voxels across keyframes.
    rolling_window : int
        How many keyframes back a track must have been updated to still be
        eligible for the rolling merge. Set to 0 to disable rolling merge.
    rolling_enabled : bool
        Master switch for the bounded rolling merge safety net.
    """

    def __init__(
        self,
        ios_thresh=0.05,
        touch_k=0,
        rolling_window=10,
        rolling_enabled=False,
    ):
        self.tracks = {}
        self.next_id = 0

        self.ios_thresh = ios_thresh
        self.touch_k = touch_k
        self.rolling_window = rolling_window
        self.rolling_enabled = rolling_enabled

        # voxel tuple -> set of track_ids containing that voxel
        self.voxel_to_tracks = {}

        self.current_frame_id = -1

    # ------------------------------------------------------------------ #
    # spatial index
    # ------------------------------------------------------------------ #
    def _index_voxels(self, track, voxels):
        tid = track.track_id
        for v in voxels:
            self.voxel_to_tracks.setdefault(v, set()).add(tid)

    def _unindex_track(self, track):
        tid = track.track_id
        for v in track.voxels:
            s = self.voxel_to_tracks.get(v)
            if s is not None:
                s.discard(tid)
                if not s:
                    del self.voxel_to_tracks[v]

    def _candidate_track_ids(self, obs):
        """Tracks with at least one voxel within Chebyshev distance k of obs."""
        cand = set()
        k = self.touch_k
        for v in obs.voxels:
            for u in dilated_neighbors(v, k):
                s = self.voxel_to_tracks.get(u)
                if s:
                    cand.update(s)
        return cand

    # ------------------------------------------------------------------ #
    # create / merge
    # ------------------------------------------------------------------ #
    def _create_track(self, obs):
        trk = ObjectTrack(self.next_id, obs)
        self.tracks[self.next_id] = trk
        self._index_voxels(trk, obs.voxels)
        self.next_id += 1
        return trk

    def _merge_tracks_into(self, primary, others):
        """Fuse ``others`` into ``primary`` and maintain the spatial index."""
        for other in others:
            if other.track_id == primary.track_id:
                continue
            self._unindex_track(other)
            new_voxels = primary.absorb_track(other)
            self._index_voxels(primary, new_voxels)
            self.tracks.pop(other.track_id, None)

    def _update_primary_with_obs(self, primary, obs):
        new_voxels = primary.absorb_observation(obs)
        self._index_voxels(primary, new_voxels)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def update_with_observations(self, observations, frame_id=None):
        """
        Incrementally ingest one keyframe's worth of observations.
        Fully online; cost per observation is independent of total track count.
        """
        if frame_id is not None:
            self.current_frame_id = max(self.current_frame_id, frame_id)

        updated_track_ids = []

        for obs in observations:
            # stamp frame id if caller didn't set one
            if obs.frame_id is None and frame_id is not None:
                obs.frame_id = frame_id
            if obs.frame_id is not None:
                self.current_frame_id = max(self.current_frame_id, obs.frame_id)

            if not obs.voxels:
                continue

            # O(1)-in-track-count candidate lookup
            cand_ids = self._candidate_track_ids(obs)

            strong = []  # (ios, trk) with ios above threshold
            for tid in cand_ids:
                trk = self.tracks.get(tid)
                if trk is None or trk.cls != obs.cls:
                    continue
                ios = compute_ios(trk.voxels, obs.voxels)
                if ios > self.ios_thresh:
                    strong.append((ios, trk))
                # NOTE: tracks that are merely *adjacent* (no real IoS overlap)
                # are intentionally ignored. Two distinct same-class objects
                # sitting next to each other (e.g. two remotes on a table)
                # must NOT be fused just because they are close.

            if strong:
                # strongest IoS becomes the primary. Other strong matches are
                # only fused if their IoS with the primary itself is also high
                # -- otherwise they are likely separate objects that happen
                # to both partially overlap this observation.
                strong.sort(key=lambda x: -x[0])
                primary = strong[0][1]
                others = []
                for _, t in strong[1:]:
                    if compute_ios(primary.voxels, t.voxels) > self.ios_thresh:
                        others.append(t)
                if others:
                    self._merge_tracks_into(primary, others)
                self._update_primary_with_obs(primary, obs)
                updated_track_ids.append(primary.track_id)

            else:
                new_trk = self._create_track(obs)
                updated_track_ids.append(new_trk.track_id)

        # bounded second pass: only between tracks updated this step and
        # other recently-updated tracks, so cost is O(|updated| * |recent|)
        if self.rolling_enabled and self.rolling_window > 0 and updated_track_ids:
            self._rolling_merge(updated_track_ids)

    def _rolling_merge(self, updated_track_ids):
        """
        Merge tracks that were updated this step with any other recent track
        of the same class whose accumulated volume overlaps or is touching.

        This catches the case where two previously-separate tracks both got
        updated by different observations in this keyframe, and their
        accumulated volumes now meet without any single observation having
        directly bridged them.
        """
        cutoff = self.current_frame_id - self.rolling_window
        recent_by_cls = {}
        for t in self.tracks.values():
            if t.last_seen_frame >= cutoff:
                recent_by_cls.setdefault(t.cls, []).append(t)

        for tid in updated_track_ids:
            a = self.tracks.get(tid)
            if a is None:
                continue
            peers = recent_by_cls.get(a.cls, [])
            for b in peers:
                if a.track_id == b.track_id or b.track_id not in self.tracks:
                    continue
                if a.track_id not in self.tracks:
                    break
                ios = compute_ios(a.voxels, b.voxels)
                if ios > self.ios_thresh:
                    self._merge_tracks_into(a, [b])

    # ------------------------------------------------------------------ #
    # post-processing
    # ------------------------------------------------------------------ #
    def finalize_geometry(self, voxel_size):
        """
        Call once SLAM is done (or periodically) to refresh per-track
        centroids and oriented bounding boxes from accumulated voxels.
        """
        for trk in self.tracks.values():
            trk.compute_geometry(voxel_size)

    # backward-compat shim for the old entry point
    def compute_all_geometry(self, global_points=None, voxel_size=None):
        if voxel_size is None:
            raise ValueError(
                "compute_all_geometry now requires voxel_size. "
                "Prefer calling finalize_geometry(voxel_size)."
            )
        self.finalize_geometry(voxel_size)
