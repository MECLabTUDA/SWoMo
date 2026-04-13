import os, math, random
import json
import glob
import cv2
from natsort import natsorted
from PIL import Image
import numpy as np
from decord import VideoReader

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FF
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from diffusers.utils import logging
import albumentations as A
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch


logger = logging.get_logger(__name__)

class SurgicalDataset(Dataset):
    def __init__(
            self,
            video_folder,
            split_mode,
            dataset_name,
            class_size=None,
            ignore_index=None,
            sample_size=256, 
            sample_n_frames=16,
            overlap_size=1,
            frame_stride=2,
            apply_augmentation=False,
            train_graph_encoder=False,
            return_graph_emb=False,
            **kwargs,
        ):
        self.split_mode = split_mode
        self.dataset_name = dataset_name
        self.class_size = class_size
        self.ignore_index = ignore_index
        self.sample_n_frames = sample_n_frames
        self.apply_augmentation = apply_augmentation
        self.train_graph_encoder = train_graph_encoder
        self.return_graph_emb = return_graph_emb
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        
        # separate transform bcs pixel_transform can transform multiple frames together.
        # graph transform with albumentation for easier mask handling
        self.pixel_rescale = transforms.Compose([
                                transforms.Resize(sample_size, antialias=None),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

        self.graph_rescale = A.Resize(sample_size[0], sample_size[1], interpolation=cv2.INTER_LANCZOS4)

        if self.apply_augmentation:
            additional_augment_targets = {}
            for i in range(1, self.sample_n_frames + 1):
                additional_augment_targets[f"image{i}"] = "image"
                additional_augment_targets[f"mask{i}"] = "mask"
                additional_augment_targets[f"cond{i}"] = "image"
            additional_augment_targets[f"cond"] = "image"

            #TODO: Think about adding cropping augmentations as well. Think throughly if would be problematic or not
            self.augment_trans = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_LANCZOS4),
                                            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                                            A.HorizontalFlip(p=0.20),
                                            A.VerticalFlip(p=0.50),
                                            A.RandomRotate90(p=0.10)],
                                            additional_targets=additional_augment_targets)
            
        video_txt = [os.path.join(vid_folder, "splits", split_mode+".txt") for vid_folder in video_folder]

        self.video_list = []
        for txt in video_txt:
            with open(txt, "r") as f:
                self.video_list.extend([os.path.join(os.path.dirname(os.path.dirname(txt)), "video_frames_jpg", x) for x in f.read().splitlines()])

        if self.dataset_name == "cataract":
            text = "sequence during cataract surgery."
        elif self.dataset_name == "cholec":
            text = "sequence during cholecystectomy surgery."

        else:
            raise NotImplementedError
        
        self.text = text
        self.sample = []
        for video in self.video_list:
            frame_path = video
            segm_path = video.replace("video_frames_jpg", "masks_sim")
            graph_path = video.replace("video_frames_jpg", "scene_graphs_sim")
            cond_video_path = video.replace("video_frames_jpg", "video_frames_sim")

            frame_list = natsorted(glob.glob(frame_path + "/*.jpg"))[::frame_stride]
            frame_list = [os.path.basename(x) for x in frame_list]
            segm_list = natsorted(glob.glob(segm_path + "/*.png"))[::frame_stride]
            graph_list = natsorted(glob.glob(graph_path + "/*.pt"))[::frame_stride]
            cond_video_list = natsorted(glob.glob(cond_video_path + "/*.jpg"))[::frame_stride]
            
            for n in range(0, len(frame_list), overlap_size):
                if n + self.sample_n_frames + 1 < len(frame_list):
                    self.sample.append({"video": os.path.basename(video),
                                        "video_folder": os.path.dirname(os.path.dirname(video)),
                                        "frame": frame_list[n:n + self.sample_n_frames + 1], # Adding one for optical flow info for graph construction under augmentation
                                        "segm": segm_list[n:n + self.sample_n_frames + 1], 
                                        "graph": graph_list[n:n + self.sample_n_frames],
                                        "cond_video": cond_video_list[n:n + self.sample_n_frames + 1]}) 

        if self.return_graph_emb:
            self.validation_graphs_loaded = []
            if 'validation_graphs' in kwargs:
                for graph in kwargs['validation_graphs']: 
                    # doing this bcs dont wanna list all 16 graphs in config file
                    all_graph = natsorted(glob.glob(os.path.dirname(graph) + "/*"))
                    index = all_graph.index(graph)
                    graphs = all_graph[index:index+self.sample_n_frames]
                    graphs = [torch.load(i) for i in graphs]
                    graphs = Batch.from_data_list(graphs)
                    self.validation_graphs_loaded.append(graphs)

        self.validation_seq_paths = []
        if 'validation_first_frames' in kwargs:
            for frame in kwargs['validation_first_frames']:
                all_frame = natsorted(glob.glob(os.path.dirname(frame) + "/*"))
                index = all_frame.index(frame)
                frames = all_frame[index:index+self.sample_n_frames]
                self.validation_seq_paths.append(frames)

        self.validation_cond_seq_paths = []
        if 'validation_cond_videos' in kwargs:
            for frame in kwargs['validation_cond_videos']:
                all_frame = natsorted(glob.glob(os.path.dirname(frame) + "/*"))
                index = all_frame.index(frame)
                frames = all_frame[index:index+self.sample_n_frames]
                self.validation_cond_seq_paths.append(frames)

    def load_scene_graph(self, graph_path):
        graph = torch.load(graph_path)
        if "x" not in graph:
            # dummy graph for the case where the graph is empty 
            G = nx.Graph()
            features = [0.0] * (self.class_size + 6)
            if self.ignore_index in ["first", "last"]:
                features[self.class_size - 1 if self.ignore_index == "last" else 0] = 1
            G.add_node(len(G.nodes), features=features, centroid=(0.0, 0.0))
            G.add_node(len(G.nodes), features=features, centroid=(0.0, 0.0))

            for idx1, data1 in G.nodes(data=True):
                for idx2, data2 in G.nodes(data=True):
                    if idx1 >= idx2:
                        continue
                    G.add_edge(idx1, idx2)
                data1['x'] = data1['features']
            graph = from_networkx(G)
        return graph

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        items = {}
        frames = []
        graphs = []
        segmentations = []
        cond_videos = []
        class_labels = []
        sequence = self.sample[index]

        if self.train_graph_encoder or self.return_graph_emb:
            if not self.apply_augmentation:
                for idx in range(self.sample_n_frames):
                    graph = self.load_scene_graph(sequence["graph"][idx])
                    graphs.append(graph)
                graphs = Batch.from_data_list(graphs)
                items["graph"] = graphs
        
            # Images and masks to pass to the graphconstructor to form graph on the fly if aumentation is applied
            else:
                graph_images = []
                graph_segmentations = []
                graph_conds = []

        # Temporary until augmentation is applied
        temp_frames = []
        temp_segmentations = []
        temp_cond_videos = []

        # Adding one for optical flow info for graph construction under augmentation
        for idx in range(self.sample_n_frames + 1):
            image = os.path.join(sequence["video_folder"], "video_frames_jpg", sequence["video"], sequence["frame"][idx])
            image = np.array(Image.open(image).convert("RGB"))
            segmentation = np.array(Image.open(sequence['segm'][idx]))
            cond_frame = np.array(Image.open(sequence['cond_video'][idx]).convert("RGB"))
            
            temp_frames.append(image)
            temp_segmentations.append(segmentation)
            temp_cond_videos.append(cond_frame)

        # Splliting this processing so that same augmentation can be applied on entire sequence instead.
        if self.apply_augmentation:
            kwargs = {"image": temp_frames[0], "mask": temp_segmentations[0], "cond": temp_cond_videos[0]}
            for idx in range(1, self.sample_n_frames + 1):
                kwargs[f"image{idx}"] = temp_frames[idx]
                kwargs[f"mask{idx}"] = temp_segmentations[idx]
                kwargs[f"cond{idx}"] = temp_cond_videos[idx]

            result = self.augment_trans(**kwargs)
            temp_frames = [result["image"]] + [result[f"image{i}"] for i in range(1, self.sample_n_frames + 1)]
            temp_segmentations = [result["mask"]] + [result[f"mask{i}"] for i in range(1, self.sample_n_frames + 1)]
            temp_cond_videos = [result["cond"]] + [result[f"cond{i}"] for i in range(1, self.sample_n_frames + 1)]

            #TODO: Using to_tensor and torch.from numpy below looks ugly but needed. Check more elegent way
            if self.train_graph_encoder or self.return_graph_emb:
                for frm, seg, cnd in zip(temp_frames, temp_segmentations, temp_cond_videos):
                    graph_images.append(FF.to_tensor(frm)) 
                    graph_segmentations.append(torch.from_numpy(seg))
                    graph_conds.append(FF.to_tensor(cnd))
                    items["graph_image"] = torch.stack(graph_images)
                    items["graph_segmentation"] = torch.stack(graph_segmentations)
                    items["graph_cond"] = torch.stack(graph_conds)

        # For now handling dataset of diffusion and graph encoder separately for simplicity
        if self.train_graph_encoder:
            for idx in range(self.sample_n_frames): 
                image, segmentation = self.graph_rescale(image=temp_frames[idx], mask=temp_segmentations[idx]).values()
                image = (image / 127.5 - 1.0).astype(np.float32)
                image = torch.from_numpy(image).permute(2, 0, 1)
                frames.append(image)

                segmentation = torch.from_numpy(segmentation.astype(np.int64))

                class_label = torch.zeros(self.class_size, dtype=torch.int64)
                class_label[segmentation.unique().long()] = 1
                class_labels.append(class_label)
                
                segmentation = F.one_hot(segmentation, num_classes=self.class_size)
                segmentation = segmentation.permute(2, 0, 1).to(dtype=torch.float32)
                segmentations.append(segmentation)

            items["image"] = torch.stack(frames)
            items["class_label"] = torch.max(torch.stack(class_labels), dim=0).values
            items["segmentation"] = torch.stack(segmentations)

        else:
            for idx in range(self.sample_n_frames):
                frames.append(temp_frames[idx])
                cond_videos.append(temp_cond_videos[idx])
            
            frames = np.array(frames)
            pixel_values = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.0
            pixel_values = self.pixel_rescale(pixel_values)

            cond_videos = np.array(cond_videos)
            cond_video_values = torch.from_numpy(cond_videos).permute(0, 3, 1, 2).contiguous()
            cond_video_values = cond_video_values / 255.0
            cond_video_values = self.pixel_rescale(cond_video_values)

            items["pixel_values"] = pixel_values
            items["cond_video_values"] = cond_video_values
            items["text"] = self.text
            items["first_frame_path"] = os.path.join(sequence["video_folder"], "video_frames_jpg", sequence["video"], sequence["frame"][0])
            items["all_frame_path"] = [os.path.join(sequence["video_folder"], "video_frames_jpg", sequence["video"], sequence["frame"][i]) for i in range(self.sample_n_frames)]
            items["cond_video_path"] = [os.path.join(sequence["video_folder"], "video_frames_sim", sequence["video"], sequence["cond_video"][i]) for i in range(self.sample_n_frames)]
        return items