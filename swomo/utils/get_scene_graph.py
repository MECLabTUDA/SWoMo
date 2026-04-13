import cv2
import numpy as np
import networkx as nx

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx, to_networkx
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

#TODO: Think about adding angle here. But also re-think orientation with 0-360, instead of just 0-180
#TODO: Think about tools that come up to down 

class GraphConstructor():
    def __init__(self, num_graph_in_3d, device, num_classes, background_label, anatomy_label, tool_label):
        self.num_graph_in_3d = num_graph_in_3d
        self.device = device
        self.num_classes = num_classes
        self.background_label = background_label
        self.anatomy_label = anatomy_label
        self.tool_label = tool_label

        self.raft_weights = Raft_Large_Weights.DEFAULT
        self.raft_transforms = self.raft_weights.transforms()
        self.raft_model = raft_large(weights=self.raft_weights, progress=True).to(self.device)
        self.raft_model.eval()

    def process_frame_mask_optical_flow(self, frame, mask):
        mask = mask.detach().cpu().numpy().astype(np.uint8)
        mask = [mask[i] for i in range(self.num_graph_in_3d)]
        frame, frame_next = self.raft_transforms(frame[:-1], frame[1:])

        with torch.no_grad():
            optical_flow = self.raft_model(frame, frame_next)
        optical_flow = optical_flow[-1]

        max_norm = torch.sum(optical_flow**2, dim=1).sqrt().max()
        epsilon = torch.finfo((optical_flow).dtype).eps
        normalized_optical_flow = optical_flow / (max_norm + epsilon)
        normalized_optical_flow = [normalized_optical_flow[i].detach().cpu().numpy() for i in range(self.num_graph_in_3d)]
        return normalized_optical_flow, mask

    def create_graph_from_mask(self,
                               mask: np.ndarray, 
                               gaussian_blur_kernel_size: int = 5,
                               apply_gaussian_blur: bool = False,
                               morph_kernel_size: int = 2,
                               touch_threshold: int = 3,
                               min_area: int = 10,
                               min_aspect_ratio: float = 0.1,
                               raft_optical_flow: np.ndarray = None,
                               ) -> nx.Graph:
        """
        Create a graph from a segmentation mask with preprocessing to handle noise.

        :param mask: Segmentation mask as a numpy array.
        :param gaussian_blur_kernel_size: Kernel size for Gaussian blur.
        :param apply_gaussian_blur: Whether to apply Gaussian blur as a noise reduction step.
        :param morph_kernel_size: Kernel size for morphological operations.
        :param touch_threshold: Minimum overlap required to consider components as touching.
        :param min_area: Minimum area for a component to be considered significant.
        :param min_aspect_ratio: Minimum aspect ratio to consider a component significant.
        :param raft_optical_flow: If provided, will include average optical flow per node.
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
        for class_id in range(0, self.num_classes):  # Assuming class IDs start from 0
            
            # Create a binary mask for the current class
            class_mask = (mask == class_id).astype(np.uint8)

            if class_id in self.background_label:
                continue
                
            # For anatomy classes, find connected components
            elif class_id in self.anatomy_label:
                num_labels, labels_im = cv2.connectedComponents(class_mask)
            
            # For tool classes, treat the entire class mask as a single component
            elif class_id in self.tool_label:
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

                one_hot_class = [0] * self.num_classes
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
        graph = from_networkx(G)
        
        # dummy graph for the case where the graph is empty 
        if "x" not in graph:
            G = nx.Graph()
            features = [0.0] * (self.num_classes + 6)
            features[self.background_label[0]] = 1
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

    def create_scene_graph(self, frame, mask):
        normalized_optical_flow, mask = self.process_frame_mask_optical_flow(frame, mask)
        graphs = []
        for msk, nof in zip(mask, normalized_optical_flow):
            graph = self.create_graph_from_mask(mask=msk,
                                                     morph_kernel_size=2,
                                                     min_area=25,
                                                     min_aspect_ratio=0.1,
                                                     raft_optical_flow=nof)
            graphs.append(graph)
        graphs = Batch.from_data_list(graphs)
        return graphs