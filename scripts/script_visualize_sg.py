import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from swomo.data.utils import get_cadis_float_cmap, get_cholecseg8k_float_cmap, get_cataract1k_float_cmap

def visualize_graph_list(data_list,
                         image_shape: tuple,
                         num_classes: int,
                         class_names,
                         node_size=200,
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
    mask_cmap = plt.cm.get_cmap(cmap, num_classes)
    class_colors = mcolors.ListedColormap(mask_cmap(np.linspace(0, 1 + 1 / num_classes, num_classes)))

    num_graphs = len(data_list)
    fig, axes = plt.subplots(1, num_graphs, figsize=(5 * num_graphs, 4), frameon=False)

    for idx, graph_data in enumerate(data_list):

        ax = axes[idx] if num_graphs > 1 else axes

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

        # ax.set_aspect(1)
        # ax.set_title(f"Graph {idx + 1}")

    # plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='png')
        plt.close()
    plt.show()


if __name__ == "__main__":

    # graph1 = torch.load("/path/to/dataset/Cholec-80/scene_graph/video01/0000003904.pt")
    # graph2 = torch.load("/path/to/dataset/Cholec-80/scene_graph/video01/0000003906.pt")
    # graph3 = torch.load("/path/to/dataset/Cholec-80/scene_graph/video01/0000003907.pt")
    # class_names = ["Black Background", "Abdominal Wall", "Liver", "Gastrointestinal Tract", "Fat", "Grasper", "Connective Tissue", "Blood", "Cystic Dust", "L-hook Electrocautery", "Gallblader", "Hepatic Vein", "Liver Ligament"]
    # palette = get_cholecseg8k_float_cmap()
    # num_classes = 13

    # graph1 = torch.load("/path/to/dataset/Cataracts-50/scene_graph/train47/0000000696.pt")
    # graph2 = torch.load("/path/to/dataset/Cataracts-50/scene_graph/train47/0000000704.pt")
    # graph3 = torch.load("/path/to/dataset/Cataracts-50/scene_graph/train47/0000000711.pt")
    # class_names = ["Pupil", "Surg. Tape", "Hand", "Eye Retr.", "Iris", "Skin", "Cornea", "Cannula", "Cap. Cystotome", "Tissue Forceps", "Primary Knife", "Ph. Handpiece", "Lens Injector", "I/A Handpiece", "Secondary Knife", "Micromanipulator", "Cap. Forceps", "Ignore"]
    # palette = get_cadis_float_cmap()
    # num_classes = 18

    graph1 = torch.load("/path/to/dataset/Cataract-1K/scene_graph/case_2613/0000000040.pt")
    graph2 = torch.load("/path/to/dataset/Cataract-1K/scene_graph/case_2613/0000000045.pt")
    graph3 = torch.load("/path/to/dataset/Cataract-1K/scene_graph/case_2613/0000000050.pt")
    class_names = ["Background", "Cornea", "Pupil", "Lens", "Slit Knife", "Gauge", "Capsulorhexis Cystotome", "Spatula", "Phacoemulsification Tip", "Irrigation-Aspiration", "Lens Injector", "Incision Knife", "Katena Forceps", "Capsulorhexis Forceps"]
    palette = get_cataract1k_float_cmap()
    num_classes = 14
    
    save_path = "graph.png"
    visualize_graph_list(data_list = [graph1, graph2, graph3],
                        image_shape = (128,160),
                        num_classes = num_classes,
                        class_names = class_names,
                        node_size=400,
                        save_path = save_path,
                        palette=palette)