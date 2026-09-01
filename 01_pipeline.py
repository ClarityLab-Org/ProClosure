import os
import numpy as np
import cv2
import yaml
import json
import torch
from pathlib import Path
import torch.multiprocessing as mp
from accelerate import Accelerator
from transformers import Sam3Processor, Sam3Model
import open3d as o3d
from dataclasses import dataclass

from mast3r_slam_main import mast3r_worker
import mast3r_slam.evaluate as eval

from object_tracker import build_global_voxel_map, GlobalObjectMap, ObjectObservation
from object_tracker import save_objects, generate_scene_graph_frames

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Run MASt3R SLAM

# -------------------------

def run_mast3r_process(args):
 
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=mast3r_worker, args=(q, args))
    p.start()
    keyframes = q.get()
    p.join()
    return keyframes

def crop_box_from_frame(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def draw_bbox(image, bbox):
    img = image.copy()
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

def save_global_wall_points(global_wall_points, save_path, video_name):
    save_path = os.path.join(save_path, f"{video_name}_wall.ply")
    if len(global_wall_points) > 0:
        structure_points_3d = np.vstack(global_wall_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(structure_points_3d)
        o3d.io.write_point_cloud(save_path, pcd)
        

@dataclass
class Mast3R_Args:
    dataset: str = ""
    config: str = ""
    save_as: str = ""
    no_viz: bool = True
    calib: str = ""

        
# -------------------------

# SAM3 + Tracking Pipeline

# -------------------------

def run_pipeline(keyframes, text_prompts, point_conf_thresh=1.5, sam3_conf_thresh=0.8):
    
    voxel_size = 0.02
    accelerator = Accelerator()
    device = accelerator.device
 
    print("[INFO] Building global voxel map...")
    voxel_map, global_points = build_global_voxel_map(keyframes, voxel_size)
 
    print("[INFO] Loading SAM3...")
    
    sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device, dtype=torch.float16)
    sam3_model.eval()
    sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
 
    obj_map = GlobalObjectMap()
    
    global_wall_points = []
    # import pdb; pdb.set_trace() 
    
    for frame_idx, kf in enumerate(keyframes):
 
        image = kf.image
        pts_map = kf.points_world.reshape(-1,3)
        pts_conf = kf.conf.reshape(-1)
        valid_conf = pts_conf > point_conf_thresh
 
        print(f"[INFO] Processing frame {frame_idx}")
 
        observations = []
 
        for text_prompt in text_prompts:
 
            inputs = sam3_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
 
            with torch.no_grad():
                outputs = sam3_model(**inputs)
 
            results = sam3_processor.post_process_instance_segmentation(outputs, threshold=0.5,
                                mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())[0]
 
            scores = results["scores"].tolist()
            boxes = results["boxes"]
            masks = results["masks"]
 
            for index in range(len(scores)):
                score = float(scores[index])
                if score < sam3_conf_thresh:
                    continue
 
                mask = masks[index].detach().cpu().numpy().astype(bool)
                mask_flat = mask.reshape(-1)
        
                finite = np.isfinite(pts_map).all(axis=1)
                nonzero = np.linalg.norm(pts_map, axis=1) > 1e-6
                keep = mask_flat & finite & nonzero & valid_conf
                pts_keep = pts_map[keep]
 
                if pts_keep.shape[0] < 20:
                    continue
            
                
                if text_prompt.lower() in ["wall", "window"]:
                    global_wall_points.append(pts_keep)
                    continue
                print("text_prompt: ", text_prompt)
                    
                bbox_2d = boxes[index].detach().cpu().numpy()
                # compute mask area
                mask_area_norm = float(mask.sum()) / float(mask.shape[0] * mask.shape[1])
                # Crop thumbnail
                thumbnail = draw_bbox(image, bbox_2d)
 
                # indices = object_points_to_indices(pts_keep,voxel_map, voxel_size)
 
                # if len(indices) < 5:
                #     continue
         
                obs = ObjectObservation(label=text_prompt, points_world=pts_keep, voxel_size=voxel_size,
                                       bbox_2d=bbox_2d, img_h=image.shape[0], img_w=image.shape[1],
                            score=score, num_points=int(pts_keep.shape[0]), mask_area_norm=mask_area_norm,
                            thumbnail=thumbnail, frame_id=frame_idx)
    
                observations.append(obs)
        # if frame_idx == 3:
        # import pdb; pdb.set_trace() 
        
        obj_map.update_with_observations(observations, frame_id=frame_idx)
        print(f"[INFO] Tracks: {len(obj_map.tracks)}")
        
    obj_map.compute_all_geometry(global_points, voxel_size)
    
    return obj_map, global_points, global_wall_points
 
     
# -------------------------

# Main

# -------------------------

if __name__ == "__main__":
    
    mp.set_start_method("spawn", force=True)
    
    config_path = "config/config_thesis.yaml"
    config_path = os.path.join(BASE_DIR, config_path)
 
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    mast3r_args = config_data["mast3r_args"]
    mast3r_args["dataset"] = os.path.join(BASE_DIR, mast3r_args["dataset"])
    mast3r_args["config"] = os.path.join(BASE_DIR, mast3r_args["config"])
    
    save_dir = config_data["logs_dir"]
    sam3_labels = config_data["sam3_labels"]
    video_name = Path(config_data["mast3r_args"]["dataset"]).stem
    # import pdb; pdb.set_trace()
                           
        
    args = Mast3R_Args(**mast3r_args)
    print("[INFO] Running MASt3R-SLAM...")
    keyframes = run_mast3r_process(args)
 
    print(f"[INFO] Received {len(keyframes)} keyframes from MASt3R.")
    obj_map, global_points, global_wall_points = run_pipeline(keyframes, sam3_labels)
    # import pdb; pdb.set_trace() 
                                     
    save_objects(save_dir, obj_map)
    save_global_wall_points(global_wall_points, save_dir, video_name)
    
    generate_scene_graph_frames(keyframes, obj_map, save_dir)
    
    print("\nFinal objects:")
    for tid, trk in obj_map.tracks.items():
        print(f"ID {tid} | class={trk.cls} | indices={len(trk.voxels)}")
 