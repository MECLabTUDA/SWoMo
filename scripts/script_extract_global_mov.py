"""
Putting together META's CoTracker to track points in a video.
Essentially using GUI to choose multiple points on first frame. 
If the number of tracking point reduces below threshold, this is flagged.
Then need to GUI again and start new annotation and start tracking from there.
Repeated itertively to get the full video tracking.

FOR NOW:
Doing some things manually. CoTracker offline also cannot take in too many frames. So deleting coupling tensors like frames that wonts used anymore. 

- Need to rewrite things that combines both offline and online !!
- Made changes here: /gris/gris-f/homestud/ssivakum/co-tracker/cotracker/predictor.py
- Made chagnes here: /gris/gris-f/homestud/ssivakum/co-tracker/cotracker/models/core/cotracker/cotracker3_offline.py
https://github.com/facebookresearch/co-tracker/issues/125
https://github.com/facebookresearch/co-tracker/issues/60
"""

#TODO: Also wanna try out the grid method on the outside region with mask or something. Probably also work better, since wont be dependenton a few points that might go missing and skew results alot!
#TODO: The mask will also enable the check with optical flow!!
#TODO: Checkout offline demo as well (more VRAM needed probably)

import os
import json
import numpy as np
from natsort import natsorted
import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
import random
import colorsys
import pandas as pd
from cotracker.predictor import CoTrackerOnlinePredictor
from typing import List, Optional, Tuple

def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
	"""Gets colormap for points."""
	colors = []
	for i in np.arange(0.0, 360.0, 360.0 / num_colors):
		hue = i / 360.0
		lightness = (50 + np.random.rand() * 10) / 100.0
		saturation = (90 + np.random.rand() * 10) / 100.0
		color = colorsys.hls_to_rgb(hue, lightness, saturation)
		colors.append((int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
	random.shuffle(colors)
	return colors


def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    colormap: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
	"""Converts a sequence of points to color code video.

	Args:
	  frames: [num_frames, height, width, 3], np.uint8, [0, 255]
	  point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
	  visibles: [num_points, num_frames], bool
	  colormap: colormap for points, each point has a different RGB color.

	Returns:
	  video: [num_frames, height, width, 3], np.uint8, [0, 255]
	"""
	num_points, num_frames = point_tracks.shape[0:2]
	if colormap is None:
		colormap = get_colors(num_colors=num_points)
	height, width = frames.shape[1:3]
	dot_size_as_fraction_of_min_edge = 0.015
	radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
	diam = radius * 2 + 1
	quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
	quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
	icon = (quadratic_y + quadratic_x) - (radius**2) / 2.0
	sharpness = 0.15
	icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)
	icon = 1 - icon[:, :, np.newaxis]
	icon1 = np.pad(icon, [(0, 1), (0, 1), (0, 0)])
	icon2 = np.pad(icon, [(1, 0), (0, 1), (0, 0)])
	icon3 = np.pad(icon, [(0, 1), (1, 0), (0, 0)])
	icon4 = np.pad(icon, [(1, 0), (1, 0), (0, 0)])

	video = frames.copy()
	for t in range(num_frames):
		# Pad so that points that extend outside the image frame don't crash us
		image = np.pad(video[t], [(radius + 1, radius + 1), (radius + 1, radius + 1), (0, 0)])
		
		for i in range(num_points):
			# The icon is centered at the center of a pixel, but the input coordinates
			# are raster coordinates.  Therefore, to render a point at (1,1) (which
			# lies on the corner between four pixels), we need 1/4 of the icon placed
			# centered on the 0'th row, 0'th column, etc.  We need to subtract
			# 0.5 to make the fractional position come out right.
			x, y = point_tracks[i, t, :] + 0.5
			x = min(max(x, 0.0), width)
			y = min(max(y, 0.0), height)

			if visibles[i, t]:
				x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
				x2, y2 = x1 + 1, y1 + 1

				# bilinear interpolation
				patch = (icon1 * (x2 - x) * (y2 - y)
				       + icon2 * (x2 - x) * (y - y1)
				       + icon3 * (x - x1) * (y2 - y)
				       + icon4 * (x - x1) * (y - y1))
				
				x_ub = x1 + 2 * radius + 2
				y_ub = y1 + 2 * radius + 2
				image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[y1:y_ub, x1:x_ub, :] + patch * np.array(colormap[i])[np.newaxis, np.newaxis, :]

			# Remove the pad
			video[t] = image[radius + 1 : -radius - 1, radius + 1 : -radius - 1].astype(np.uint8)
	return video


def preprocess_frames(frames):
	"""Preprocess frames to model inputs."""
	frames = frames.float()
	frames = frames / 255 * 2 - 1
	return frames


def inference(frames, query_points, model):
	"""Preprocess video to match model inputs format."""
	frames = preprocess_frames(frames)
	query_points = query_points.float()
	frames, query_points = frames[None], query_points[None]

	outputs = model(frames, query_points)
	tracks, occlusions, expected_dist = (
		outputs['tracks'][0],
		outputs['occlusion'][0],
		outputs['expected_dist'][0])

	# Binarize occlusions
	visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
	return tracks, visibles


def estimate_rigid_transform(P0, Pt, R0):
    """
    Estimate rotation R and translation T such that:
        Pt ≈ R @ P0 + T
    where P0 and Pt are Nx2 arrays of tracked points in frame 0 and t.
    """
    # Center the points
    centroid_0 = np.mean(P0, axis=0)
    centroid_t = np.mean(Pt, axis=0)
    P0_centered = P0 - centroid_0
    Pt_centered = Pt - centroid_t

    # Compute rotation using SVD
    H = P0_centered.T @ Pt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure right-handed coordinate system (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    T = centroid_t - R @ centroid_0
    Rt = R @ R0 + T
    return Rt


def moving_average(points, window_size=3):
    """Compute moving average of points with a specified window size to smooth our jerky tracking."""
    smoothed = []
    for i in range(len(points)):
        start = max(0, i - window_size + 1)
        window = points[start:i+1]
        smoothed.append(np.mean(window, axis=0))
    return np.array(smoothed)


def exponential_moving_average(points, alpha=0.2):
    """Compute exponential moving average of points to smooth our jerky tracking."""
    smoothed = [points[0]]
    for i in range(1, len(points)):
        new_point = alpha * points[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(new_point)
    return np.array(smoothed)


if __name__ == "__main__":
	
	# video_dir = "/path/to/Cataract-1K/videos"
	# tracking_ann_dir = "/path/to/ann/Cataract-1K/ann_glob_pretracking_points"
	# save_ann_dir = "/path/to/ann/Cataract-1K/ann_glob_posttracking_points"
	# save_vis_dir = "/path/to/ann/Cataract-1K/ann_glob_posttracking_points_vis"
	# refinement_ann_path = "/path/to/ann/Cataract-1K/ann_glob_posttracking_refinement.csv"
	# checkpoint_path = "/path/to/src/co-tracker/checkpoints/scaled_online.pth"
	
	video_dir = "/path/to/Cataracts-50/videos"
	tracking_ann_dir = "/path/to/ann/Cataracts-50/ann_glob_pretracking_points"
	save_ann_dir = "/path/to/ann/Cataracts-50/ann_glob_posttracking_points"
	save_vis_dir = "/path/to/ann/Cataracts-50/ann_glob_posttracking_points_vis"
	refinement_ann_path = "/path/to/ann/Cataracts-50/ann_glob_posttracking_refinement.csv"
	checkpoint_path = "/path/to/src/co-tracker/checkpoints/scaled_online.pth"

	threshold_points = 17
	threshold_consecutive_missing = 5

	colormap = get_colors(200)
	device = torch.device('cuda')
	refinement_ann_df = pd.DataFrame(columns=['video_id', 'frame_idx'])

	# Switch between all videos or only the ones that need refinement
	video_list = natsorted(os.listdir(tracking_ann_dir))
	video_list = video_list[0:1]
	# video_list = pd.read_csv(refinement_ann_path, skipinitialspace=True)['video_id'].unique().tolist()
	# video_list = [v + ".json" for v in video_list]
	
	for video_name in video_list:
		video_path = os.path.join(video_dir, video_name.replace(".json", ".mp4"))
		tracking_ann_path = os.path.join(tracking_ann_dir, video_name)
		save_ann_path = os.path.join(save_ann_dir, video_name.replace(".json", ".csv"))
		save_vis_path = os.path.join(save_vis_dir, video_name.replace(".json", ".mp4"))

		glob_ann_df = pd.DataFrame(columns=['frame_idx', 'glob_x', 'glob_y'])
		video = media.read_video(video_path)
		video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

		with open(tracking_ann_path, 'r') as f:
			tracking_points = json.load(f)

		# number of frames with tracking point ann
		tracking_frame = tracking_points['frames']
		
		refinement_flag = False
		video_viz = []
		tracks_list = []
		visibles_list = []
		glob_ref_point_list = []

		for f_idx in range(len(tracking_frame)):
			if f_idx == 0:
				start_idx = tracking_frame[f_idx]['frame_idx']
				if start_idx != 0:
					print(f"First frame does not have tracking points for video: {video_name}")
					print(f"Use future global point")
					video_viz.append(video[:start_idx])
					for t in range(start_idx):
						r0 = np.array(tracking_frame[f_idx]['global_point'])
						glob_ref_point_list.append(r0)

			if len(tracking_frame) == 1:
				start_idx = tracking_frame[f_idx]['frame_idx']
				end_idx = video_tensor.shape[1]
			else:
				start_idx = tracking_frame[f_idx]['frame_idx']
				end_idx = tracking_frame[f_idx + 1]['frame_idx'] if f_idx + 1 < len(tracking_frame) else video_tensor.shape[1]
				
			video_idx = video_tensor[:, start_idx:end_idx, :, :]
			num_frames = video_idx.shape[1]
			select_points = np.array(tracking_frame[f_idx]['tracking_points'])
			select_points = np.insert(select_points, 0, 0, axis=1)
			select_points =  torch.from_numpy(select_points).to(device=device, dtype=torch.float32).unsqueeze(0)

			model = CoTrackerOnlinePredictor(checkpoint=checkpoint_path, offline=True)
			model = model.to(device)
			model = model.eval()

			window_frames = []
			tracks, visibles = None, None
			is_first_step = True
			step = model.step
		
			for i, frame in enumerate(video_idx[0]):
				if i % step == 0 and i != 0:
					video_chunk = torch.stack(window_frames[-step * 2 :]).unsqueeze(0).to(device=device)
					pred_tracks, pred_visibility = model(video_chunk, queries=select_points, is_first_step=is_first_step)
					is_first_step = False

				window_frames.append(frame)
		
			# processing final video frames, when length is not a multiple of model.step
			fin_window_frames = window_frames[-(i % step) - step - 1 :]
			video_chunk = torch.stack(fin_window_frames[-step * 2 :]).unsqueeze(0).to(device=device)
			pred_tracks, pred_visibility = model(video_chunk[-step * 2 :], queries=select_points, is_first_step=is_first_step)
			
			tracks = pred_tracks.to(dtype = torch.float64).squeeze(0).permute(1, 0, 2).detach().cpu().numpy()
			visibles = pred_visibility.squeeze(0).permute(1, 0).detach().cpu().numpy()
			video_viz.append(paint_point_track(video[start_idx:end_idx,:,:,:], tracks, visibles, colormap))

			if refinement_flag is False:
				for lacking_frame_idx in range(num_frames):
					num_points = visibles[:,lacking_frame_idx].sum()
					
					if num_points < threshold_points:
						num_points_consecutive = visibles[:, lacking_frame_idx:lacking_frame_idx + threshold_consecutive_missing].sum()
						if num_points_consecutive < threshold_points * threshold_consecutive_missing:
							# Skip if there is already a refinement in the future
							ann_idx = [i['frame_idx'] for i in tracking_frame]
							if ann_idx[-1] < start_idx + lacking_frame_idx:
								print(f"Lacking tracking points for video: {video_name} at frame idx: {start_idx + lacking_frame_idx}")
								refinement_flag = True
								if ann_idx[-1] + 16 < start_idx + lacking_frame_idx:
									refinement_ann_df.loc[len(refinement_ann_df)] = [video_name.split(".")[0], start_idx + lacking_frame_idx]
								else:
									print("Future is not far enought!!!")
									refinement_ann_df.loc[len(refinement_ann_df)] = [video_name.split(".")[0], ann_idx[-1] + 16]
								break
						
			glob_ref_point = []
			for t in range(num_frames):  
				if t == 0: 
					r0 = np.array(tracking_frame[f_idx]['global_point'])
					glob_ref_point.append(r0)
		
				else:
					# find the points that are visible in both the first frame and the current frame
					joint_visibles = visibles[:, 0] & visibles[:, t]
					indices = np.where(joint_visibles)[0]
		
					# if no points are visible, directly use the last reference point
					if len(indices) == 0:
						glob_ref_point.append(glob_ref_point[-1])
						continue
				
					p0 = np.stack([tracks[i, 0, :] for i in indices])
					pt = np.stack([tracks[i, t, :] for i in indices])
					rt = estimate_rigid_transform(p0, pt, r0)
					glob_ref_point.append(rt)

			glob_ref_point_list.extend(glob_ref_point)
		
		# smooth out the reference point
		glob_ref_points = np.array(glob_ref_point_list)
		glob_ref_points = np.stack(glob_ref_points)
		glob_ref_points = moving_average(glob_ref_points)

		for i, pt in enumerate(glob_ref_points):
			glob_ann_df.loc[len(glob_ann_df)] = [int(i), round(pt[0]), round(pt[1])]

		tracks_ref = glob_ref_points[np.newaxis, :, :]
		visibles_ref = np.ones((1, video_tensor.shape[1]), dtype=bool)
		video_viz = np.concatenate(video_viz, axis=0)
		video_viz = paint_point_track(video_viz, tracks_ref, visibles_ref, [(255,255,255)])

		refinement_ann_df.to_csv(refinement_ann_path, index=False)	
		glob_ann_df.to_csv(save_ann_path, index=False)

		print(f"Completing the tracking for: {video_name}")
		if save_vis_path is not None:
			media.write_video(save_vis_path, video_viz, fps=8)