import os
import glob
import cv2
import json
import numpy as np
import networkx as nx
from tqdm import tqdm
from PIL import Image, ImageColor
from natsort import natsorted
from joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

import torch
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from torchvision.io import write_jpeg
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

#TODO: Think about adding angle here. But also re-think orientation with 0-360, instead of just 0-180

def visualize_graph_list(data_list,
                         image_shape: tuple,
                         num_classes: int,
                         class_names,
                         node_size=400,
                         save_path: str = None,
                         palette=None):
    """
    Visualizes a list of graphs with nodes colored by their class, keeping colors consistent for class IDs.

    Args:
        data_list (list[Data]): A list of PyTorch Geometric Data objects representing the graphs to visualize.
        image_shape (tuple): The shape of the original image (height, width).
        num_classes (int): The number of different classes for coloring nodes.
        class_names (list[str]): A list of class names corresponding to each class.
        save_path (str, optional): Path to save the visualized graph. If None, the graph is not saved.
        palette: The palette to use for coloring nodes based on their class.
    """

    # Create a color map for the nodes based on their class. Default to 'viridis' if no cmap provided.
    if palette is None:
        cmap = 'viridis'
    else:
        cmap = LinearSegmentedColormap.from_list('CustomCMAP', palette.numpy()[:num_classes - 1])  # w/o ignore
    mask_cmap = matplotlib.colormaps.get_cmap(cmap).resampled(num_classes)
    class_colors = mcolors.ListedColormap(mask_cmap(np.linspace(0, 1 + 1 / num_classes, num_classes)))

    num_graphs = len(data_list)
    fig, axes = plt.subplots(1, num_graphs, figsize=(5 * num_graphs, 4), frameon=False)

    for idx, graph_data in enumerate(data_list):

        ax = axes[idx] if num_graphs > 1 else axes
        
        if graph_data.num_nodes > 0:
            # Convert to NetworkX graph if necessary
            if isinstance(graph_data, Data):
                graph = to_networkx(graph_data.cpu(), to_undirected=True)
                # Manually transfer node features from PyG Data to NetworkX
                for node, data in enumerate(graph_data.x.cpu()):
                    graph.nodes[node]['features'] = data
            else:
                graph = graph_data

            try:
                pos = {node: (data['features'][-5] * image_shape[1], (1 - data['features'][-4]) * image_shape[0]) for node, data in
                    graph.nodes(data=True)}
            except KeyError:
                pos = {node: (data['x'][-5] * image_shape[1], (1 - data['x'][-4]) * image_shape[0]) for
                    node, data in
                    graph.nodes(data=True)}

            # Add labels to the nodes (e.g., class name and node ID)
            for node, (x, y) in pos.items():
                features = graph.nodes[node]['features']
                class_id = np.argmax(features[:num_classes])  # Extract class ID from one-hot vector  # w/o ignore
                class_name = class_names[class_id]  # Get class name based on class ID
                node_color = class_colors(class_id / num_classes)  # Map class ID to color
                nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=node_size, node_color=[node_color], ax=ax)
                ax.text(x, y, f'{class_name}', fontsize=20, ha='right', va='center')

            # Draw edges
            nx.draw_networkx_edges(graph, pos, edge_color="grey", alpha=0.3, width=1.8, ax=ax)

        else:
            ax.text(0.5, 0.5, 'No nodes in graph', fontsize=20, ha='center', va='center')

    # plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='jpg')
        plt.close()


def create_graph_from_mask(mask: np.ndarray, 
                           num_classes: int,
                           background_label: list = None,
                           anatomy_label: list = None,
                           tool_label: list = None,
                           gaussian_blur_kernel_size: int = 5,
                           apply_gaussian_blur: bool = False,
                           morph_kernel_size: int = 2,
                           touch_threshold: int = 3,
                           min_area: int = 10,
                           min_aspect_ratio: float = 0.1,
                           raft_optical_flow: np.ndarray = None,
                           save_dir: str = None,
                           visualize_sg: bool = False,
                           visualize_of: bool = False,
                           class_names: list = None,
                           palette: torch.Tensor = None,
                           vis_optical_flow: torch.Tensor = None
                           ) -> nx.Graph:
    """
    Create a graph from a segmentation mask with preprocessing to handle noise.

    :param mask: Segmentation mask as a numpy array.
    :param num_classes: Total number of classes, including background.
    :param background_label: Label of the background class, if any.
    :param anatomy_label: Label of the anatomy classes, if any.
    :param tool_label: Label of the tool classes, if any.
    :param gaussian_blur_kernel_size: Kernel size for Gaussian blur.
    :param apply_gaussian_blur: Whether to apply Gaussian blur as a noise reduction step.
    :param morph_kernel_size: Kernel size for morphological operations.
    :param touch_threshold: Minimum overlap required to consider components as touching.
    :param min_area: Minimum area for a component to be considered significant.
    :param min_aspect_ratio: Minimum aspect ratio to consider a component significant.
    :param raft_optical_flow: If provided, will include average optical flow per node.
    :param save_dir: If provided, will save the graph as a torch.geometric .pt file.
    :param visualize_sg: If true, will save a visualization of the scene graph.
    :param visualize_of: If true, will save a visualization of the optical flow.
    :param class_names: List of class names for visualization.
    :param palette: Color palette for visualization.
    :param vis_optical_flow: Non-processed optical flow tensor for visualization.
    :return: A NetworkX graph representing the segmented objects and their relationships.
    """

    # Convert the mask to a numpy array if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Pre-process mask to reduce impact of annotation noise
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)  # Convert mask to uint8 if it's not already

    if apply_gaussian_blur:
        mask = cv2.GaussianBlur(mask, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)

    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    # Erosion followed by dilation to remove small components
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    G = nx.Graph()
    image_height, image_width = mask.shape

    components = {}

    # First pass: identify components and add nodes
    for class_id in range(0, num_classes):  # Assuming class IDs start from 0
        
        # Create a binary mask for the current class
        class_mask = (mask == class_id).astype(np.uint8)

        if class_id in background_label:
            continue
            
        # For anatomy classes, find connected components
        elif class_id in anatomy_label:
            num_labels, labels_im = cv2.connectedComponents(class_mask)
        
        # For tool classes, treat the entire class mask as a single component
        elif class_id in tool_label:
            num_labels = 2
            labels_im = class_mask     

        else:
            continue 

        for i in range(1, num_labels):
            component_mask = (labels_im == i).astype(np.uint8)

            # Calculate component properties
            ys, xs = np.where(component_mask)
            if len(xs) == 0 or len(ys) == 0:  # Skip empty components
                continue
            area = len(xs)
            x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
            aspect_ratio = (y_max - y_min + 1) / (x_max - x_min + 1)

            if area < min_area or aspect_ratio < min_aspect_ratio:
                continue  # Skip components that don't meet the criteria

            relative_width = (x_max - x_min) / image_width
            relative_height = (y_max - y_min) / image_height
            relative_centroid_x = xs.mean() / image_width
            relative_centroid_y = ys.mean() / image_height

            one_hot_class = [0] * num_classes
            one_hot_class[class_id] = 1
            features = one_hot_class + [relative_width, relative_height, relative_centroid_x, relative_centroid_y]

            if raft_optical_flow is not None:
                optical_flow_x_values = raft_optical_flow[0, :, :][component_mask == 1]
                optical_flow_y_values = raft_optical_flow[1, :, :][component_mask == 1]
                optical_flow_x_mean = np.mean(optical_flow_x_values)
                optical_flow_y_mean = np.mean(optical_flow_y_values)
                features = one_hot_class + [relative_width, relative_height, relative_centroid_x, relative_centroid_y, optical_flow_x_mean, optical_flow_y_mean]

            # Add node to the graph
            node_id = len(G.nodes)  # Unique ID for the node
            G.add_node(node_id, features=features, centroid=(relative_centroid_x, relative_centroid_y))
            components[node_id] = component_mask

    # Second pass: add edges between touching components
    for idx1, data1 in G.nodes(data=True):

        for idx2, data2 in G.nodes(data=True):
            if idx1 >= idx2:
                continue

            # Use individual component masks for each node
            component1_mask = components[idx1]
            component2_mask = components[idx2]

            # Check if the dilated components are touching
            dilated_component1 = cv2.dilate(component1_mask, np.ones((3, 3), np.uint8), iterations=1)
            touching = cv2.bitwise_and(dilated_component1, component2_mask)

            # Add an edge if components are touching
            if np.sum(touching) >= touch_threshold:
                G.add_edge(idx1, idx2)

        data1['x'] = data1['features']

    if save_dir is not None:
        torch.save(from_networkx(G), save_dir)

    if visualize_sg:
        vis_save_path = os.path.join(os.path.dirname(os.path.dirname(save_dir))+"_vis", os.path.basename(os.path.dirname(save_dir)), os.path.basename(save_dir).replace('.pt', '.jpg'))
        os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
        visualize_graph_list(data_list = [from_networkx(G)],
                            image_shape = (128,128),
                            num_classes = num_classes,
                            class_names = class_names,
                            save_path = vis_save_path,
                            palette = palette)

    if visualize_of:  
        of_vis_path = vis_save_path.replace('scene_graphs_', 'optical_flow_')
        flow_image = flow_to_image(vis_optical_flow).to("cpu")
        write_jpeg(flow_image, of_vis_path)

    return G


class SurgicalDataset(Dataset):
    def __init__(self,
                 data_root,
                 video_prefix,
                 video_id,
                 size=128
                 ):

        self.data_root = data_root
        self.video_prefix = video_prefix
        self.video_id = video_id
        self.size = size

        real_frame_list = natsorted(glob.glob(os.path.join(data_root, "video_frames_jpg", video_prefix+video_id, "*.jpg")))
        self.real_mask_list = [i.replace('.jpg', '.png').replace('video_frames_jpg', 'masks_real') for i in real_frame_list]
        self.real_graph_list = [i.replace('.jpg', '.pt').replace('video_frames_jpg', 'scene_graphs_real') for i in real_frame_list]

        sim_frame_list = [i.replace('video_frames_jpg', 'video_frames_sim') for i in real_frame_list]
        self.sim_mask_list = [i.replace('masks_real', 'masks_sim') for i in self.real_mask_list]
        self.sim_graph_list = [i.replace('scene_graphs_real', 'scene_graphs_sim') for i in self.real_graph_list]
        
        # last pair is duplicate of last frame, to compensate length and lack of optical flow and depth
        self.real_frame_pairs = [(real_frame_list[i], real_frame_list[i+1]) for i in range(len(real_frame_list) - 1)]
        self.real_frame_pairs.append((real_frame_list[-1], real_frame_list[-1]))
        self.sim_frame_pairs = [(sim_frame_list[i], sim_frame_list[i+1]) for i in range(len(sim_frame_list) - 1)]
        self.sim_frame_pairs.append((sim_frame_list[-1], sim_frame_list[-1]))
        self._length = len(self.real_frame_pairs)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        real_frame_path, real_frame_next_path = self.real_frame_pairs[i]
        example["real_frame_path"] = real_frame_path
        example["real_frame_next_path"] = real_frame_next_path
        example["real_mask_path"] = self.real_mask_list[i]
        example["real_graph_path"] = self.real_graph_list[i]

        sim_frame_path, sim_frame_next_path = self.sim_frame_pairs[i]
        example["sim_frame_path"] = sim_frame_path
        example["sim_frame_next_path"] = sim_frame_next_path
        example["sim_mask_path"] = self.sim_mask_list[i]
        example["sim_graph_path"] = self.sim_graph_list[i]

        real_frame = Image.open(example["real_frame_path"]).convert('RGB')
        real_frame = F.to_tensor(real_frame)
        real_frame = F.resize(real_frame, size=[self.size, self.size], antialias=False)
        example["real_frame"] = real_frame
        real_frame_next = Image.open(example["real_frame_next_path"]).convert('RGB')
        real_frame_next = F.to_tensor(real_frame_next)
        real_frame_next = F.resize(real_frame_next, size=[self.size, self.size], antialias=False)
        example["real_frame_next"] = real_frame_next

        sim_frame = Image.open(example["sim_frame_path"]).convert('RGB')
        sim_frame = F.to_tensor(sim_frame)
        sim_frame = F.resize(sim_frame, size=[self.size, self.size], antialias=False)
        example["sim_frame"] = sim_frame
        sim_frame_next = Image.open(example["sim_frame_next_path"]).convert('RGB')
        sim_frame_next = F.to_tensor(sim_frame_next)
        sim_frame_next = F.resize(sim_frame_next, size=[self.size, self.size], antialias=False)
        example["sim_frame_next"] = sim_frame_next

        real_mask = np.array(Image.open(example["real_mask_path"])).astype(np.uint8)
        example["real_mask"] = cv2.resize(real_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        sim_mask = np.array(Image.open(example["sim_mask_path"])).astype(np.uint8)
        example["sim_mask"] = cv2.resize(sim_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return example


def build_loaders(batch, data_root, video_prefix, video_id, size):
    dataset = SurgicalDataset(
        data_root=data_root,
        video_prefix=video_prefix,
        video_id=video_id,
        size=size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        num_workers=32,
        shuffle=False,
    )
    return dataloader


if __name__ == "__main__":

    batch_size = 16
    device = "cuda"
    size = 256
    num_classes = 14
    background_label = [0]
    anatomy_label = [1, 2]
    tool_label = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    label_info = "/path/to/ann/Cataract-1K/ann_tool_classes.json"
    data_root = "/path/to/Cataract-1K" #/gris/scratch-gris-filesrv
    video_prefix = "case_"

    # label_info = "/path/to/ann/Cataracts-50/ann_tool_classes.json"
    # data_root = "/path/to/Cataracts-50" #/gris/scratch-gris-filesrv
    # video_prefix = "train"

    video_list = natsorted(glob.glob(os.path.join(data_root, "video_frames_jpg", video_prefix + "*")))
    
    inf_start = int(os.environ.get("INF_START"))
    inf_end = int(os.environ.get("INF_END"))
    video_list = video_list[inf_start:inf_end]

    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=True).to(device)
    raft_model.eval()

    with open(label_info, "r") as f:
        class_info = json.load(f)
    palette = []
    class_names = []
    for cls in class_info:
        color = ImageColor.getrgb(class_info[cls]["color"])
        palette.append(color)
        class_names.append(class_info[cls]["name"])
    palette = torch.from_numpy(np.asarray(palette))/255.0
    class_names = ["Background", "Cornea", "Pupil", "Lens", "Slit Knife", "Gauge", "Capsulorhexis Cystotome", "Spatula", "Phacoemulsification Tip", "Irrigation-Aspiration", "Lens Injector", "Incision Knife", "Katena Forceps", "Capsulorhexis Forceps"]

    for video in video_list:
        print(f"Processing video: {os.path.basename(video)}")
        real_sg_save_dir = video.replace("video_frames_jpg", "scene_graphs_real")
        sim_sg_save_dir = video.replace("video_frames_jpg", "scene_graphs_sim")
        if not os.path.exists(real_sg_save_dir):
            os.makedirs(real_sg_save_dir, exist_ok=True)
        if not os.path.exists(sim_sg_save_dir):
            os.makedirs(sim_sg_save_dir, exist_ok=True)
        video_id = os.path.basename(video).replace(video_prefix, "")

        dataloader = build_loaders(batch_size, data_root, video_prefix, video_id, size)
        tqdm_object = tqdm(dataloader, total=len(dataloader))
        
        for batch in tqdm_object:
            batch = {k: (v if k.endswith("path") else v.to(device)) for k, v in batch.items()}
            if len(batch["real_mask"]) != batch_size: process_batch_size = len(batch["real_mask"])
            else: process_batch_size = batch_size

            real_mask_batch = [batch["real_mask"][i] for i in range(process_batch_size)]
            real_frame_batch, real_frame_next_batch = raft_transforms(batch["real_frame"], batch["real_frame_next"])
            sim_mask_batch = [batch["sim_mask"][i] for i in range(process_batch_size)]
            sim_frame_batch, sim_frame_next_batch = raft_transforms(batch["sim_frame"], batch["sim_frame_next"])
            
            with torch.no_grad():
                real_optical_flow = raft_model(real_frame_batch, real_frame_next_batch)
                sim_optical_flow = raft_model(sim_frame_batch, sim_frame_next_batch)  
            
            def process_optical_flow(optical_flow):
                optical_flow = optical_flow[-1]
                max_norm = torch.sum(optical_flow**2, dim=1).sqrt().max()
                epsilon = torch.finfo((optical_flow).dtype).eps
                normalized_optical_flow = optical_flow / (max_norm + epsilon)
                normalized_optical_flow = [normalized_optical_flow[i].detach().cpu().numpy() for i in range(process_batch_size)]
                return normalized_optical_flow, optical_flow
            
            real_normalized_optical_flow, real_of = process_optical_flow(real_optical_flow)
            sim_normalized_optical_flow, sim_of = process_optical_flow(sim_optical_flow)

            # for i, (mask, graph_path) in enumerate(zip(real_mask_batch, batch["real_graph_path"])):
            #     create_graph_from_mask(
            #         mask,
            #         num_classes=num_classes,
            #         background_label=background_label,
            #         anatomy_label=anatomy_label,
            #         tool_label=tool_label,
            #         morph_kernel_size=2,
            #         min_area=25,
            #         min_aspect_ratio=0.1,
            #         save_dir=graph_path,
            #         raft_optical_flow=real_normalized_optical_flow[i],
            #         class_names=class_names,
            #         palette=palette,
            #         vis_optical_flow=real_of[i])

            # for i, (mask, graph_path) in enumerate(zip(sim_mask_batch, batch["sim_graph_path"])):
            #     create_graph_from_mask(
            #         mask,
            #         num_classes=num_classes,
            #         background_label=background_label,
            #         anatomy_label=anatomy_label,
            #         tool_label=tool_label,
            #         morph_kernel_size=2,
            #         min_area=25,
            #         min_aspect_ratio=0.1,
            #         save_dir=graph_path,
            #         raft_optical_flow=sim_normalized_optical_flow[i],
            #         class_names=class_names,
            #         palette=palette,
            #         vis_optical_flow=sim_of[i])


            Parallel(n_jobs=process_batch_size)(delayed(create_graph_from_mask)(mask,
                                                                        num_classes=num_classes,
                                                                        background_label=background_label,
                                                                        anatomy_label=anatomy_label,
                                                                        tool_label=tool_label,
                                                                        morph_kernel_size=2,
                                                                        min_area=25,
                                                                        min_aspect_ratio=0.1,
                                                                        save_dir=graph_path,
                                                                        raft_optical_flow=real_normalized_optical_flow[i],
                                                                        class_names=class_names,
                                                                        palette=palette,
                                                                        vis_optical_flow=real_of[i])
                                                                        for i, (mask, graph_path) in enumerate(zip(real_mask_batch, batch["real_graph_path"])))

            Parallel(n_jobs=process_batch_size)(delayed(create_graph_from_mask)(mask,
                                                                        num_classes=num_classes,
                                                                        background_label=background_label,
                                                                        anatomy_label=anatomy_label,
                                                                        tool_label=tool_label,
                                                                        morph_kernel_size=2,
                                                                        min_area=25,
                                                                        min_aspect_ratio=0.1,
                                                                        save_dir=graph_path,
                                                                        raft_optical_flow=sim_normalized_optical_flow[i],
                                                                        class_names=class_names,
                                                                        palette=palette,
                                                                        vis_optical_flow=sim_of[i]) 
                                                                        for i, (mask, graph_path) in enumerate(zip(sim_mask_batch, batch["sim_graph_path"])))    