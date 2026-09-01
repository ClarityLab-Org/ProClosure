import pathlib
import numpy as np
import json
import cv2
import os


def save_objects(out_dir, obj_map):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # objects_dir = out_dir / "objects"
    # objects_dir.mkdir(exist_ok=True, parents=True)

    thumbs_dir = out_dir / "thumbnails"
    thumbs_dir.mkdir(exist_ok=True, parents=True)

    centroids = []
    obbs = []

    metadata = {}

    for gid in sorted(obj_map.tracks.keys()):

        trk = obj_map.tracks[gid]

        # centroid
        if trk.centroid_world is not None:
            centroid = trk.centroid_world.astype(np.float32)
        else:
            centroid = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        # 3D bounding box
        if trk.obb_3d is not None:
            obb = trk.obb_3d
            obb_dict = {"center": obb.center.tolist(), "extent": obb.extent.tolist(),
                        "R": obb.R.tolist()}
        else:
            obb_dict = None

        # save thumbnail
        thumb_path = None
        if trk.thumbnail is not None:
            thumb_path = thumbs_dir / f"{gid:03d}.png"
            cv2.imwrite(str(thumb_path), trk.thumbnail)

        # metadata
        metadata[int(gid)] = [trk.cls, trk.owner, centroid.tolist(), obb_dict,
                               trk.bbox_2d.tolist(), str(thumb_path)]

    # save metadata
    with open(out_dir / "thumbnails_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved {len(metadata)} objects")


COLOR_PALETTE = [
    {"name": "red", "rgb": (255, 0, 0)},
    {"name": "green", "rgb": (0, 255, 0)},
    {"name": "blue", "rgb": (0, 0, 255)},
    {"name": "yellow", "rgb": (255, 255, 0)},
    {"name": "magenta", "rgb": (255, 0, 255)},
    {"name": "cyan", "rgb": (0, 255, 255)},
    {"name": "orange", "rgb": (255, 165, 0)},
    {"name": "purple", "rgb": (128, 0, 128)},
    {"name": "brown", "rgb": (165, 42, 42)},
]


def generate_scene_graph_frames(keyframes, obj_map, out_dir):

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    relationship_dir = os.path.join(out_dir, "relationship_images")
    relationship_dir = pathlib.Path(relationship_dir)
    relationship_dir.mkdir(exist_ok=True, parents=True)

    frame_objects = {}
    json_data = {}

    # collect objects per frame
    for gid, trk in obj_map.tracks.items():
        for f in trk.visible_frames:
            frame_objects.setdefault(f, []).append(gid)

    for frame_id, obj_ids in frame_objects.items():

        if len(obj_ids) < 2:
            continue

        frame = keyframes[frame_id].image.copy()
        frame_key = f"frame_{frame_id:04d}.png"
        json_data[frame_key] = []

        for oid in obj_ids:

            trk = obj_map.tracks[oid]
            if frame_id not in trk.frame_bboxes:
                continue

            bbox = trk.frame_bboxes[frame_id]
            x1, y1, x2, y2 = map(int, bbox)
            color_idx = oid % len(COLOR_PALETTE)
            color_info = COLOR_PALETTE[color_idx]
            color_rgb = color_info["rgb"]

            # OpenCV expects BGR
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

            json_data[frame_key].append({
                "obj_id": oid,
                "class": trk.cls,
                "color_name": color_info["name"],
                "color_id": color_idx,
                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
            })

        cv2.imwrite(str(relationship_dir / frame_key), frame)

    # save JSON legend
    with open(out_dir / "relations_prep.json", "w") as f:
        json.dump(json_data, f, indent=2)
