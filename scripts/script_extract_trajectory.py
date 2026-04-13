import os
import glob
import cv2
import json
import torch
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import astropy.units as u
from fil_finder import FilFinder2D
from natsort import natsorted
from joblib import Parallel, delayed
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
from scipy.ndimage import convolve, label
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

#TODO: Right now changed things to 512x288. This makes sense for trajectory

def extract_traj_from_mask(mask_anatomy: np.ndarray,
                           mask_tool: np.ndarray,
                           frame: torch.Tensor = None,
                           num_classes: int = None,
                           label_id: dict = None,
                           traj_size: tuple = None,
                           background_label: list = None,
                           anatomy_label: list = None,
                           tool_label: list = None,
                           gaussian_blur_kernel_size: int = 5,
                           apply_gaussian_blur: bool = False,
                           morph_kernel_size: int = 2,
                           global_pos: list = None,
                           ref_frame_idx: list = None,
                           save_dir_traj: str = None,
                           visualize: bool = False):
    """
    Extract trajectory infos from image & segmentation mask with preprocessing to handle noise.

    :param mask_anatomy: Segmentation mask of anatomy as a numpy array.
    :param mask_tool: Segmentation mask of tool as a numpy array.
    :param frame: Frame image as a numpy array, used for visualization.
    :param num_classes: Total number of classes, including background.
    :param label_id: Dictionary mapping class names to their respective IDs.
    :param traj_size: Size of the image for trajectory extraction (width, height).
    :param background_label: List containing background label.
    :param anatomy_label: List containing anatomy labels.
    :param tool_label: List containing tool labels.
    :param gaussian_blur_kernel_size: Kernel size for Gaussian blur.
    :param apply_gaussian_blur: Whether to apply Gaussian blur as a noise reduction step.
    :param morph_kernel_size: Kernel size for morphological operations.
    :param global_pos: List containing global x and y position.
    :param ref_frame_idx: Index of the reference frame for extracting eye size model.
    :param save_dir_traj: If provided, will save the trajectory as a JSON file.
    :param visualize: If True, saves a visualization of the trajectory on the frame.
    """

    # Convert the mask to a numpy array if it's a tensor
    mask_anatomy, mask_tool = [
        x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        for x in (mask_anatomy, mask_tool)]
    
    # Convert mask to uint8 if it's not already
    mask_anatomy = mask_anatomy.astype(np.uint8, copy=False)
    mask_tool = mask_tool.astype(np.uint8, copy=False)

    # Pre-process mask to reduce impact of annotation noise
    if apply_gaussian_blur:
        mask_anatomy = cv2.GaussianBlur(mask_anatomy, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)
        mask_tool = cv2.GaussianBlur(mask_tool, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)
    
    # Erosion followed by dilation to remove small components
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask_anatomy = cv2.morphologyEx(mask_anatomy, cv2.MORPH_OPEN, kernel)
    mask_tool = cv2.morphologyEx(mask_tool, cv2.MORPH_OPEN, kernel)

    image_height, image_width, _ = frame.shape
    frame = cv2.cvtColor(frame.cpu().numpy(), cv2.COLOR_RGB2BGR) if isinstance(frame, torch.Tensor) else frame
    frame = cv2.resize(frame, traj_size, interpolation=cv2.INTER_LANCZOS4)
    trajectory = {}

    # Add global position
    trajectory["global_pos"] = global_pos
    trajectory["ref_frame_idx"] = ref_frame_idx
    cv2.circle(frame, global_pos, 3, (255,0,0), -1, lineType=cv2.LINE_AA)

    for class_id in range(0, num_classes):
        if class_id in background_label:
            continue

        elif class_id in anatomy_label:
            # Resize frame for trajectory extraction
            class_mask = (mask_anatomy == class_id).astype(np.uint8)
            traj_class_mask = cv2.resize(class_mask, traj_size, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(traj_class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            try:
                merged_points = np.vstack(contours).squeeze()
                ellipse = cv2.fitEllipse(merged_points)
            except:
                ellipse = ((0,0), (100,100), 90)

            # Save anatomy trajectory information
            trajectory[class_id] = {
                "class_name": label_id[str(class_id)]["name"],
                "centroid": [round(ellipse[0][0]), round(ellipse[0][1])],
                "length": [round(ellipse[1][0]), round(ellipse[1][1])],
                "angle": round(ellipse[2], 2),
            }
            frame = cv2.ellipse(frame, ellipse, (255, 255, 0), 2)

        elif class_id in tool_label:
            # Resize for trajectory extraction
            tip, tip_2, o_angle, o_angle_2, length = None, None, None, None, None
            class_mask = (mask_tool == class_id).astype(np.uint8)
            traj_class_mask = cv2.resize(class_mask, traj_size, interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(traj_class_mask)
            mask_points = np.column_stack((xs, ys)).astype(np.float32)
            
            if len(mask_points) == 0:
                continue
            
            polished_mask_tool = cv2.GaussianBlur(traj_class_mask, (7, 7), 0)
            polished_mask_tool = cv2.morphologyEx(polished_mask_tool, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            ysp, xsp = np.where(polished_mask_tool)
            polished_mask_points = np.column_stack((xsp, ysp)).astype(np.float32)
            if len(polished_mask_points) == 0:
                continue

            skeleton_mask = skeletonize(traj_class_mask).astype(np.uint8)
            ysk, xsk = np.where(skeleton_mask > 0)
            skeleton_points = np.column_stack((xsk, ysk))
        
            pca = PCA(n_components=2)
            pca.fit(polished_mask_points)
            vec = pca.components_[0].astype(np.float32)
            vec /= np.linalg.norm(vec)
            if vec[0] < 0:
                vec = -vec
            xv, yv = vec

            # For tools that are just straight
            if class_id in [4, 8, 11]:
                # Find centroid and find point that is furthest from centroid but to the upper side
                centroid = mask_points.mean(axis=0)
                upper_points = mask_points[mask_points[:, 1] < centroid[1]]
                dist = np.linalg.norm(upper_points - centroid, axis=1)
                
                o_angle = np.degrees(np.arctan2(-yv, xv)) % 180
                tip = upper_points[np.argmax(dist)]
                tip_end = (tip[0] - 1000 * math.cos(math.radians(o_angle)), tip[1] + 1000 * math.sin(math.radians(o_angle)))
                cv2.line(frame, (round(tip[0]), round(tip[1])), (round(tip_end[0]), round(tip_end[1])), (0, 255, 50), 2)
                
            # For tools that are bend
            elif class_id in [5, 6, 7]:
                # Prune the uneccasary branches, and only keep longest path
                try:
                    fil = FilFinder2D(skeleton_mask, beamwidth=0.0*u.pix, mask=skeleton_mask)
                    fil.medskel(verbose=False)
                    fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=5 * u.pix, prune_criteria='length')
                    skeleton_mask = fil.skeleton_longpath.astype(np.uint8)
                except:
                    skeleton_mask = skeleton_mask
                ysk, xsk = np.where(skeleton_mask > 0)
                skeleton_points = np.column_stack((xsk, ysk)) 

                # Find endpoints from the skeleton
                if len(skeleton_points) == 0:
                    continue
                top_idx = np.argmin(skeleton_points[:, 1])
                bottom_idx = np.argmax(skeleton_points[:, 1])
                start_pt = skeleton_points[top_idx]
                end_pt = skeleton_points[bottom_idx]
                line = end_pt - start_pt

                # Find distance of each skeleton point to the straight line
                # If the furthest point is above a threshold, consider it a bend
                distances = []
                for pt in skeleton_points:
                    t = np.clip(np.dot(pt - start_pt, line) / np.dot(line, line), 0, 1)
                    projection = start_pt + t * line
                    dist = np.linalg.norm(pt - projection)
                    distances.append(dist)

                distances = np.array(distances)
                max_dist = distances.max()
                bend_pt = skeleton_points[np.argmax(distances)]
                start2bend_dist = np.linalg.norm(bend_pt - start_pt)

                # If bend point exceed bend_threshold from straight line, consider it a bend
                # If the distance between start point and the bend point exceed threshold 
                bend_threshold = 8
                start2bend_threshold = 35

                if max_dist < bend_threshold and start2bend_dist < start2bend_threshold:
                    # No bend - one straight line
                    tip = start_pt
                    o_angle = np.degrees(np.arctan2(-(bend_pt[1]-start_pt[1]), (bend_pt[0]-start_pt[0]))) % 180
                    cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
                else:
                    # One bend - two straight lines
                    tip = start_pt
                    tip_2 = bend_pt
                    o_angle = np.degrees(np.arctan2(-(bend_pt[1]-start_pt[1]), (bend_pt[0]-start_pt[0]))) % 180
                    o_angle_2 = np.degrees(np.arctan2(-(end_pt[1]-bend_pt[1]), (end_pt[0]-bend_pt[0]))) % 180
                    piecewise_line = np.array([start_pt, bend_pt, end_pt])
                    cv2.polylines(frame, [piecewise_line], False, (0, 0, 255), 2)

            # For katena forcep with big opening
            elif class_id in [12]:
                o_angle = np.degrees(np.arctan2(-yv, xv)) % 180
                kernel = np.array([[1,1,1], [1,10,1], [1,1,1]])
                conv = convolve(skeleton_mask.astype(int), kernel, mode='constant', cval=0)

                endpoints_mask = (conv == 11)
                branchpoints_mask = (conv >= 13)
                branch_pt = np.argwhere(branchpoints_mask)
                end_pt = np.argwhere(endpoints_mask)

                # If there is a branchpoint, find the highest branchpoint
                if len(branch_pt) > 0:
                    highest_branch_pt = branch_pt.min(axis=0)
                    # Find furthest endpoint in pca direction
                    projections = [np.dot(ep - highest_branch_pt, vec) for ep in end_pt]
                    max_proj_idx = np.argsort(projections)
                    
                    furthest_pt = end_pt[max_proj_idx[0]]
                    cv2.circle(frame, (furthest_pt[1], furthest_pt[0]), 3, (0,255,0), -1)
                    length = 0
                    tip = (furthest_pt[1], furthest_pt[0])
                    
                    # Check if second furthest point is sufficiently far
                    # Shows opening of forcep. And new tip is average of in between of two
                    if len(max_proj_idx) > 1:
                        if projections[max_proj_idx[1]] < -20:
                            second_furthest_pt = end_pt[max_proj_idx[1]]
                            cv2.circle(frame, (second_furthest_pt[1], second_furthest_pt[0]), 3, (0,255,0), -1)
                            second_tip = (second_furthest_pt[1], second_furthest_pt[0])
                            length = np.linalg.norm(np.array(tip) - np.array(second_tip))
                            tip = (np.sum((tip, second_tip), axis=0) / 2).astype(np.uint)

                # If no branchpoint, just find highest endpoint, from each disconnected component
                else:
                    tip = []
                    skeleton_mask_dilated = cv2.dilate(skeleton_mask, np.ones((2,2), np.uint8))
                    labeled_skeleton, num_labels = label(skeleton_mask_dilated)
                    for i in range(1, num_labels+1):
                        component_mask = (labeled_skeleton == i)
                        component_endpoints = np.argwhere(endpoints_mask & component_mask)
                        if len(component_endpoints) != 0:
                            highest_idx = np.argmin(component_endpoints[:,0])
                            furthest_pt = component_endpoints[highest_idx]
                            tip.append((furthest_pt[1], furthest_pt[0]))

                    # If there are multiple tips, find the highest two tips
                    if len(tip) == 1:
                        cv2.circle(frame, (tip[0]), 3, (0,255,0), -1)
                        length = 0
                        tip = tip[0]

                    elif len(tip) == 2:
                        cv2.circle(frame, (tip[0]), 3, (0,255,0), -1)
                        cv2.circle(frame, (tip[1]), 3, (0,255,0), -1)
                        length = np.linalg.norm(np.array(tip[0]) - np.array(tip[1]))
                        tip = (np.sum((tip[0], tip[1]), axis=0) / 2).astype(np.uint)

                    elif len(tip) > 2:
                        sorted_idx = np.argsort([pt[1] for pt in tip])
                        tip = [tip[i] for i in sorted_idx[:2]]
                        cv2.circle(frame, (tip[0]), 3, (0,255,0), -1)
                        cv2.circle(frame, (tip[1]), 3, (0,255,0), -1)
                        length = np.linalg.norm(np.array(tip[0]) - np.array(tip[1]))
                        tip = (np.sum((tip[0], tip[1]), axis=0) / 2).astype(np.uint)

                if length > 0:
                    length_start = (tip[0] - (length / 2) * math.cos(math.radians(o_angle)), tip[1] - (length / 2) * math.sin(math.radians(o_angle)))
                    length_end = (tip[0] + (length / 2) * math.cos(math.radians(o_angle)), tip[1] + (length / 2) * math.sin(math.radians(o_angle)))
                    cv2.line(frame, (round(length_start[0]), round(length_start[1])), (round(length_end[0]), round(length_end[1])), (255, 0, 0), 2)
                tip_end = (tip[0] - 1000 * math.cos(math.radians(o_angle)), tip[1] + 1000 * math.sin(math.radians(o_angle)))
                cv2.line(frame, (round(tip[0]), round(tip[1])), (round(tip_end[0]), round(tip_end[1])), (0, 255, 50), 2)

            # For cap forcep with small opening
            #TODO: Cleanup, quickly written this part
            elif class_id in [13]:
                rect = cv2.minAreaRect(mask_points)
                # The longer side is the height, the shorter side is the width
                o_centroid_x, o_centroid_y = rect[0][0], rect[0][1]
                o_height, o_width = (rect[1][0], rect[1][1]) if rect[1][0] > rect[1][1] else (rect[1][1], rect[1][0])
                o_angle = rect[2] + 90 if rect[1][0] < rect[1][1] else rect[2]
                o_angle = (-o_angle) % 180
                
                # Using the middle of bounding box as tip
                tip = []
                if o_angle <= 90:
                    tip.append(o_centroid_x+(o_height/2)*(math.cos(math.radians(o_angle))))
                else:
                    tip.append(o_centroid_x-(o_height/2)*(math.cos(math.radians(180-o_angle))))   
                tip.append(o_centroid_y-(o_height/2)*(math.sin(math.radians(o_angle))))

                # Split bounding box along the height into n_component
                # For each split do connected component analysis and find distance between two centroids
                n_component = 20
                c_height = o_height / n_component
                box_pts = cv2.boxPoints(rect).astype(np.float32)
                L0 = np.linalg.norm(box_pts[1] - box_pts[0])
                L1 = np.linalg.norm(box_pts[2] - box_pts[1])

                # Choose the longer edge as the height axis
                if L0 >= L1:
                    long_vec = (box_pts[1] - box_pts[0]) / (L0 + 1e-12)
                else:
                    long_vec = (box_pts[2] - box_pts[1]) / (L1 + 1e-12)

                ux, uy = long_vec[0], long_vec[1]

                box_center = np.array([o_centroid_x, o_centroid_y], dtype=np.float32)
                tip_vec = np.array([tip[0], tip[1]], dtype=np.float32) - box_center
                if np.dot(tip_vec, np.array([ux, uy], dtype=np.float32)) < 0:
                    ux, uy = -ux, -uy

                vx, vy = -uy, ux
                hw = o_width / 2.0
                hh = c_height / 2.0

                length = []
                for n in range(n_component):
                    offset = (n - (n_component - 1) / 2.0) * c_height
                    c_centroid_x = o_centroid_x + offset * ux
                    c_centroid_y = o_centroid_y + offset * uy
                    p1 = (c_centroid_x + ux*hh + vx*hw, c_centroid_y + uy*hh + vy*hw)
                    p2 = (c_centroid_x + ux*hh - vx*hw, c_centroid_y + uy*hh - vy*hw)
                    p3 = (c_centroid_x - ux*hh - vx*hw, c_centroid_y - uy*hh - vy*hw)
                    p4 = (c_centroid_x - ux*hh + vx*hw, c_centroid_y - uy*hh + vy*hw)

                    new_box = np.array([p1, p2, p3, p4], dtype=np.int32)
                    new_box_mask = cv2.fillPoly(np.zeros_like(traj_class_mask, dtype=np.uint8), [new_box], 1)
                    cropped_mask = cv2.bitwise_and(traj_class_mask, new_box_mask)
                    num_labels, labels_im = cv2.connectedComponents(cropped_mask)
                    centroids = []

                    if num_labels == 3:
                        for i in range(1, num_labels):
                            component_mask = (labels_im == i).astype(np.uint8)
                            cys, cxs = np.where(component_mask)
                            centroids.append((cxs.mean(), cys.mean()))
                        length.append(np.linalg.norm(np.array(centroids[0]) - np.array(centroids[1])))

                # Using pca direction for angle
                o_angle = np.degrees(np.arctan2(-yv, xv)) % 180
                tip_end = (tip[0] - 1000 * math.cos(math.radians(o_angle)), tip[1] + 1000 * math.sin(math.radians(o_angle)))
                cv2.line(frame, (round(tip[0]), round(tip[1])), (round(tip_end[0]), round(tip_end[1])), (0, 255, 50), 2)
                
                # Only if 8 out n_component have 2 connected components, add length line
                if length is not None and len(length) > 8:
                    length = np.mean(length)
                    length_start = (tip[0] - (length / 2) * math.cos(math.radians(o_angle)), tip[1] - (length / 2) * math.sin(math.radians(o_angle)))
                    length_end = (tip[0] + (length / 2) * math.cos(math.radians(o_angle)), tip[1] + (length / 2) * math.sin(math.radians(o_angle)))
                    cv2.line(frame, (round(length_start[0]), round(length_start[1])), (round(length_end[0]), round(length_end[1])), (255, 0, 0), 2)
                else: 
                    length = 0
                
            else:
                rect = cv2.minAreaRect(mask_points)
                # The longer side is the height, the shorter side is the width
                o_centroid_x, o_centroid_y = rect[0][0], rect[0][1]
                o_height, o_width = (rect[1][0], rect[1][1]) if rect[1][0] > rect[1][1] else (rect[1][1], rect[1][0])
                o_angle = rect[2] + 90 if rect[1][0] < rect[1][1] else rect[2]
                o_angle = (-o_angle) % 180

                tip = []
                if o_angle <= 90:
                    tip.append(o_centroid_x+(o_height/2)*(math.cos(math.radians(o_angle))))
                else:
                    tip.append(o_centroid_x-(o_height/2)*(math.cos(math.radians(180-o_angle))))   
                tip.append(o_centroid_y-(o_height/2)*(math.sin(math.radians(o_angle))))

                o_angle = np.degrees(np.arctan2(-yv, xv)) % 180
                tip_end = (tip[0] - 1000 * math.cos(math.radians(o_angle)), tip[1] + 1000 * math.sin(math.radians(o_angle)))
                cv2.line(frame, (round(tip[0]), round(tip[1])), (round(tip_end[0]), round(tip_end[1])), (0, 255, 50), 2)
                

            trajectory[class_id] = {
                "class_name": label_id[str(class_id)]["name"],
                "tip": [round(tip[0]), round(tip[1])],
                "angle": round(o_angle, 2)
            }
            if tip_2 is not None and o_angle_2 is not None:
                trajectory[class_id]["tip_2"] = [round(tip_2[0]), round(tip_2[1])]
                trajectory[class_id]["angle_2"] = round(o_angle_2, 2)
            if length is not None:
                trajectory[class_id]["length"] = round(length)
        
    if save_dir_traj is not None:
        with open(save_dir_traj, 'w') as f:
            json.dump(trajectory, f, indent=2)

    if visualize:
        sav_dir_vis = save_dir_traj.replace("ann_trajectories", "ann_trajectories_vis").replace("json", "jpg")
        os.makedirs(os.path.dirname(sav_dir_vis), exist_ok=True)
        cv2.imwrite(sav_dir_vis, frame)



class SurgicalDataset(Dataset):
    def __init__(self,
                 data_root,
                 video_prefix,
                 video_id
                 ):

        self.data_root = data_root
        self.video_prefix = video_prefix
        self.video_id = video_id

        frame_list = natsorted(glob.glob(os.path.join(data_root, "video_frames_jpg", video_prefix+video_id, "*.jpg")))
        self.mask_list_anatomy = [i.replace('.jpg', '.png').replace('video_frames_jpg', 'masks_anatomy') for i in frame_list]
        self.mask_list_tool = [i.replace('.jpg', '.png').replace('video_frames_jpg', 'masks_tool') for i in frame_list]
        self.traj_list = [i.replace('.jpg', '.json').replace('video_frames_jpg', 'ann_trajectories') for i in frame_list]
        self.global_pos = os.path.dirname(frame_list[0]).replace('video_frames_jpg', 'ann_global_pos') + ".csv"
        self.global_pos = pd.read_csv(self.global_pos)
        self.ref_frame_idx = os.path.dirname(frame_list[0]).replace('video_frames_jpg', 'ann_tracking_points') + ".json"
        self.ref_frame_idx = json.load(open(self.ref_frame_idx, 'r'))['frames'][0]["frame_idx"]
        # last pair is duplicate of last frame, to compensate length and lack of optical flow and depth
        self.frame_pairs = [(frame_list[i], frame_list[i+1]) for i in range(len(frame_list) - 1)]
        self.frame_pairs.append((frame_list[-1], frame_list[-1]))
        self._length = len(self.frame_pairs)


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        frame_path, frame_next_path = self.frame_pairs[i]
        example["frame_path"] = frame_path
        example["frame_next_path"] = frame_next_path
        example["mask_anatomy_path"] = self.mask_list_anatomy[i]
        example["mask_tool_path"] = self.mask_list_tool[i]
        example["traj_path"] = self.traj_list[i]

        frame = Image.open(example["frame_path"])
        example["frame_np"] = np.array(frame)

        mask_anatomy = np.array(Image.open(example["mask_anatomy_path"])).astype(np.uint8)
        # mask_anatomy = cv2.resize(mask_anatomy, self.size, interpolation=cv2.INTER_NEAREST)
        example["mask_anatomy"] = mask_anatomy
        mask_tool = np.array(Image.open(example["mask_tool_path"])).astype(np.uint8)
        # mask_tool = cv2.resize(mask_tool, self.size, interpolation=cv2.INTER_NEAREST)
        example["mask_tool"] = mask_tool

        glob_pos = self.global_pos.iloc[i].to_dict()
        glob_pos = [glob_pos['glob_x'], glob_pos['glob_y']]
        example["global_pos"] = glob_pos
        example["ref_frame_idx"] = self.ref_frame_idx
        return example


def build_loaders(batch, data_root, video_prefix, video_id):
    dataset = SurgicalDataset(
        data_root=data_root,
        video_prefix=video_prefix,
        video_id=video_id
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        num_workers=16,
        shuffle=False,
    )
    return dataloader


if __name__ == "__main__":
    batch_size = 16
    device = "cuda"
    traj_size = (512, 288) # Used for trajectory extraction

    data_root = "/path/to/Cataract-1K" #/gris/scratch-gris-filesrv
    label_ann = "/path/to/ann/Cataract-1K/ann_tool_classes.json"
    video_prefix = "case_"

    # data_root = "/path/to/Cataracts-50" #/gris/scratch-gris-filesrv
    # label_ann = "/path/to/ann/Cataracts-50/ann_tool_classes.json"
    # video_prefix = "train"
    
    num_classes = 14
    background_label = [0]
    anatomy_label = [1, 2]
    tool_label = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    with open(label_ann, "r") as f:
        label_id = json.load(f)

    video_list = natsorted(glob.glob(os.path.join(data_root, "video_frames_jpg", video_prefix + "*")))
    video_list = video_list[1000:]

    for video in video_list:
        print(f"Processing video: {os.path.basename(video)}")
        video_id = os.path.basename(video).replace(video_prefix, "")
        target_traj_dir = video.replace("video_frames_jpg", "ann_trajectories")
        os.makedirs(target_traj_dir, exist_ok=True)
        
        dataloader = build_loaders(batch_size, data_root, video_prefix, video_id)
        tqdm_object = tqdm(dataloader, total=len(dataloader))
        for batch in tqdm_object:
            batch = {k: (v if k.endswith("path") or k.endswith("global_pos") else v.to(device)) for k, v in batch.items()}

            if len(batch["mask_anatomy"]) != batch_size: process_batch_size = len(batch["mask_anatomy"])
            else: process_batch_size = batch_size
            
            mask_anatomy_batch = [batch["mask_anatomy"][i] for i in range(process_batch_size)]
            mask_tool_batch = [batch["mask_tool"][i] for i in range(process_batch_size)]
            
            global_pos_batch = [[int(batch["global_pos"][0][i]), int(batch["global_pos"][1][i])] for i in range(process_batch_size)]
            ref_frame_idx_batch = [int(i) for i in list(batch["ref_frame_idx"].cpu().numpy())]
            
            # Parallel(n_jobs=process_batch_size)(delayed(extract_traj_from_mask)(mask=mask,
            #                                                                               frame=frame,
            #                                                                               num_classes=num_classes,
            #                                                                               background_label=background_label,
            #                                                                               cornea_label=cornea_label,
            #                                                                               pupil_label=pupil_label,
            #                                                                               morph_kernel_size=2,
            #                                                                               min_area=25,
            #                                                                               min_aspect_ratio=0.1,
            #                                                                               save_dir_graph=graph_path,
            #                                                                               save_dir_traj=traj_path,
            #                                                                               midas_monocular_depth=monocular_depth[i],
            #                                                                               raft_optical_flow=normalized_optical_flow[i]) 
            #                                                                               for i, (mask, frame, graph_path, traj_path) in enumerate(zip(mask_batch, frame_batch, batch["graph_path"], batch["traj_path"])))

            for i, (mask_anatomy, mask_tool, frame, global_pos, ref_frame_idx, traj_path) in enumerate(zip(mask_anatomy_batch, mask_tool_batch, batch["frame_np"], global_pos_batch, ref_frame_idx_batch, batch["traj_path"])):
                extract_traj_from_mask(mask_anatomy=mask_anatomy,
                                       mask_tool=mask_tool,
                                       frame=frame,
                                       num_classes=num_classes,
                                       label_id=label_id,
                                       traj_size=traj_size,
                                       background_label=background_label,
                                       anatomy_label=anatomy_label,
                                       tool_label=tool_label,
                                       morph_kernel_size=2,
                                       global_pos=global_pos,
                                       ref_frame_idx=ref_frame_idx,
                                       save_dir_traj=traj_path,
                                       visualize=True)