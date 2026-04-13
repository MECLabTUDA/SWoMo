import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv, SAGEConv, ChebConv, GINConv, global_mean_pool
from torch_geometric.nn import GraphNorm, BatchNorm, LayerNorm, InstanceNorm
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx, to_dense_adj

def weighted_binary_cross_entropy(output: torch.Tensor,
                                  target: torch.Tensor,
                                  positive_weight: float = 1.0,
                                  negative_weight: float = 1.0,
                                  reduction: str = 'mean') -> torch.Tensor:
    """
    Computes a weighted binary cross-entropy loss.

    Args:
        output (torch.Tensor): The predicted probabilities.
        target (torch.Tensor): The target labels.
        positive_weight (float, optional): The weight for positive samples. Default is 1.0.
        negative_weight (float, optional): The weight for negative samples. Default is 1.0.
        reduction (str, optional):

    Returns:
        torch.Tensor: The mean weighted binary cross-entropy loss.
    """
    bce_loss = F.binary_cross_entropy(output, target, reduction='none')
    weights = target * positive_weight + (1 - target) * negative_weight
    weighted_bce_loss = weights * bce_loss
    if reduction == 'mean':
        return weighted_bce_loss.mean()
    elif reduction == 'sum':
        return weighted_bce_loss.sum()
    else:
        raise ValueError


#TODO!!: Rename this bcs not GCN anymore.
class GCNGraphEncoder(nn.Module):
    """
    GraphEncoder encodes a graph's node features into a latent space.
    """

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int, dropout: float = 0.5, graph_conv_type: str = "GCNConv", graph_norm_type: str = "GroupNorm", global_pooling: bool = False):
        """
        Initialize the GraphEncoder.
        :param input_dim: Dimension of input features per node.
        :param hidden_dim: Dimension of hidden layer.
        :param z_dim: Dimension of the output latent space (split into mu and logvar).
        :param dropout: Dropout rate to use after each layer.
        :param graph_conv_type: Type of graph convolutional layer to use.
        :param graph_norm_type: Type of graph normalization layer to use.
        :param global_pooling: Whether to use global pooling for the final latent space.
        """
        super(GCNGraphEncoder, self).__init__()
        self.global_pooling = global_pooling
        self.conv1 = self._get_conv_layer(graph_conv_type, input_dim, hidden_dim)
        self.bn1 = self._get_norm_layer(graph_norm_type, hidden_dim)
        self.conv2 = self._get_conv_layer(graph_conv_type, hidden_dim, hidden_dim)
        self.bn2 = self._get_norm_layer(graph_norm_type, hidden_dim)
        self.conv3 = self._get_conv_layer(graph_conv_type, hidden_dim, hidden_dim)
        self.bn3 = self._get_norm_layer(graph_norm_type, hidden_dim)
        self.conv4 = self._get_conv_layer(graph_conv_type, hidden_dim, z_dim)
        self.dropout_rate = dropout

    def _get_norm_layer(self, norm_type, dim):
        """
        Dynamically return the normalization layer.
        :param norm_type: Type of normalization.
        :param dim: Number of features (channels) in the layer.
        :return: The appropriate normalization layer.
        """
        if norm_type == "GraphNorm":
            return GraphNorm(dim)
        elif norm_type == "BatchNorm":
            return BatchNorm(dim)
        elif norm_type == "LayerNorm":
            return LayerNorm(dim)  
        elif norm_type == "InstanceNorm":
            return InstanceNorm(dim)
        else:
            raise ValueError(f"Invalid norm_type '{norm_type}'.")
        
    def _get_conv_layer(self, conv_type, input_dim, output_dim):
        """
        Dynamically return the normalization layer.
        :param norm_type: Type of normalization.
        :param dim: Number of features (channels) in the layer.
        :return: The appropriate normalization layer.
        """
        if conv_type == "GCNConv":
            return GCNConv(input_dim, output_dim)
        elif conv_type == "GATConv":
            return GATConv(input_dim, output_dim, heads=4, concat=False)
        elif conv_type == "GATv2Conv":
            return GATv2Conv(input_dim, output_dim, heads=4, concat=False)
        elif conv_type == "TransformerConv":
            return TransformerConv(input_dim, output_dim, heads=4, concat=False)
        elif conv_type == "SAGEConv":
            return SAGEConv(input_dim, output_dim)
        elif conv_type == "ChebConv":
            return ChebConv(input_dim, output_dim, K=3)
        elif conv_type == "GINConv":
            mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )
            return GINConv(mlp)


        else:
            raise ValueError(f"Invalid conv_type '{conv_type}'.")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the encoder.
        :param x: Node features of shape [num_nodes, input_dim].
        :param edge_index: Edge indices of shape [2, num_edges].
        :return: Latent space representation of the graph.
        """
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv4(x, edge_index)  # No activation function here, directly outputting the latent space
        if self.global_pooling:
            x = global_mean_pool(x, batch)
        return x


class GraphDecoder(nn.Module):
    """
    GraphDecoder decodes latent variables into reconstructed node features.
    """

    def __init__(self, z_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the GraphDecoder.
        :param z_dim: Dimension of the latent space.
        :param hidden_dim: Dimension of hidden layer.
        :param output_dim: Dimension of the reconstructed node features.
        """
        super(GraphDecoder, self).__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor):
        """
        Forward pass of the decoder.
        :param z: Latent variables of shape [num_nodes, z_dim].
        :return: Reconstructed node features.
        """
        z = torch.leaky_relu(self.linear1(z))
        return torch.sigmoid(self.linear2(z)), None


class GraphRNNDecoder(nn.Module):
    """
    Autoregressive Graph Decoder for generating graphs with variable number of nodes.

    Attributes:
        z_dim (int): Dimension of the latent space vector.
        hidden_dim (int): Dimension of the hidden layers in the RNN.
        node_feature_dim (int): Dimension of the node feature vector.
        max_nodes (int): Maximum number of nodes that can be generated for a graph.
    """

    def __init__(self, z_dim: int, hidden_dim: int, node_feature_dim: int, max_nodes: int):
        """
        Initializes the GraphDecoder with specified dimensions and maximum node count.

        Args:
            z_dim (int): Dimension of the latent space vector.
            hidden_dim (int): Dimension of the hidden layers in the RNN.
            node_feature_dim (int): Dimension of the node feature vector.
            max_nodes (int): Maximum number of nodes that can be generated for a graph.
        """
        super(GraphRNNDecoder, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim
        self.max_nodes = max_nodes

        # Initial transformation of z
        self.initial_dense = nn.Linear(z_dim, hidden_dim)

        # RNN for autoregressive generation
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Output layers for node features and stopping probability
        self.node_features = nn.Linear(hidden_dim, node_feature_dim)
        self.stop_prob = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor):
        """
        Forward pass for generating graph nodes from a latent space vector.

        Args:
            z (torch.Tensor): A batch of latent space vectors (shape: [batch_size, z_dim]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Node features tensor (shape: [batch_size, num_generated_nodes, node_feature_dim]).
                - Stop probabilities tensor (shape: [batch_size, num_generated_nodes, 1]).
        """
        # Prepare initial hidden state
        h = torch.relu(self.initial_dense(z)).unsqueeze(0)  # (1, batch_size, hidden_dim)

        # Prepare inputs for RNN (initially just zeros)
        x = torch.zeros(z.size(0), 1, self.hidden_dim).to(z.device)

        nodes = []
        stop_probs = []

        for _ in range(self.max_nodes):
            # Generate next node features and update hidden state
            x, h = self.rnn(x, h)

            # Decide features of the new node
            node_feat = self.node_features(x.squeeze(1))
            nodes.append(node_feat)

            # Decide whether to stop
            stop = torch.sigmoid(self.stop_prob(x.squeeze(1)))
            stop_probs.append(stop)

            if stop.mean() > 0.5:  # If on average more than half the batch wants to stop
                break

        nodes = torch.stack(nodes, dim=1)  # (batch_size, num_generated_nodes, node_feature_dim)
        stop_probs = torch.stack(stop_probs, dim=1)  # (batch_size, num_generated_nodes, 1)

        return nodes, stop_probs


class NodeFeatureDecoder(nn.Module):
    """
    Decoder for reconstructing node features in a Graph Variational Autoencoder.
    """

    def __init__(self, z_dim: int, hidden_dim: int, feature_dim: int, dropout: float = 0.5):
        """
        Initializes the NodeFeatureDecoder with specified dimensions.

        Args:
            z_dim (int): Dimension of the latent space vector.
            hidden_dim (int): Dimension of the hidden layers in the decoder.
            feature_dim (int): Dimension of the node feature vector (number of features per node).
            dropout (float): Dropout rate to use after each layer for regularization.
        """
        super(NodeFeatureDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, feature_dim)
        self.dropout_rate = dropout

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the node feature decoder.

        Args:
            z (torch.Tensor): The latent node representations (shape: [num_nodes, z_dim]).

        Returns:
            torch.Tensor: The reconstructed node features (shape: [num_nodes, feature_dim]).
        """
        h = F.leaky_relu(self.bn1(self.fc1(z)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.leaky_relu(self.bn2(self.fc2(h)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        reconstructed_features = torch.sigmoid(self.fc3(h))  # Using sigmoid for binary features
        return reconstructed_features


class NodeFeaturePooledDecoder(nn.Module):
    """
    Decoder for reconstructing node features in a Graph Variational Autoencoder from pooled encodings.
    """

    def __init__(self, z_dim: int, hidden_dim: int, feature_dim: int, dropout: float = 0.5, num_of_nodes: int = 30):
        """
        Initializes the NodeFeaturePooledDecoder with specified dimensions.

        Args:
            z_dim (int): Dimension of the latent space vector.
            hidden_dim (int): Dimension of the hidden layers in the decoder.
            feature_dim (int): Dimension of the node feature vector (number of features per node).
            num_of_nodes (int): Number of nodes per graph (after zero-padding).
            dropout (float): Dropout rate to use after each layer for regularization.
        """
        super(NodeFeaturePooledDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(1, num_of_nodes)
        self.bn2 = nn.BatchNorm1d(num_of_nodes)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the node feature decoder.

        Args:
            z (torch.Tensor): The latent node representations (shape: [batch, z_dim]).

        Returns:
            torch.Tensor: The reconstructed node features (shape: [batch * num_of_nodes, feature_dim]).
        """
        h = F.leaky_relu(self.bn1(self.fc1(z)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        h = F.leaky_relu(self.fc2(h.unsqueeze(-1)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        h = h.permute(0, 2, 1).reshape(-1, self.hidden_dim)

        h = F.leaky_relu(self.bn3(self.fc3(h)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        reconstructed_features = torch.sigmoid(self.fc4(h))  # Using sigmoid for binary features
        return reconstructed_features
    

class InnerProductDecoder(nn.Module):
    """
    Inner Product Decoder for Graph Variational Autoencoders (VGAE).

    This decoder reconstructs the graph's adjacency matrix by computing the inner
    product between latent node representations. It assumes the probability of an edge
    between two nodes can be modeled as the sigmoid of the inner product of their
    corresponding latent representations.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the inner product decoder.

        Args:
            z (torch.Tensor): The latent node representations of shape (batch_size, num_nodes, latent_dim),
                              where batch_size is typically 1 for a single graph.

        Returns:
            torch.Tensor: The reconstructed adjacency matrix of the graph of shape (num_nodes, num_nodes).
                          Values represent the probabilities of edges between nodes.
        """
        # Compute the inner product between all pairs of node latent representations
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
        # adj_pred = torch.tanh(torch.matmul(z, z.t()))

        return adj_pred


class MLPPooledDecoder(nn.Module):
    """
    MLP-based decoder for reconstructing the adjacency matrix from a pooled encoding in a Graph Variational Autoencoder.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.5, num_of_nodes: int = 30):
        """
        Initializes the MLPDecoder with specified dimensions and dropout rate.

        Args:
            input_dim (int): Dimension of the input latent node representations.
            hidden_dim (int): Dimension of the hidden layer in the decoder.
            num_of_nodes (int): Number of nodes per graph (after zero-padding).
            dropout (float): Dropout rate to use after the activation function.
        """
        super(MLPPooledDecoder, self).__init__()
        self.dropout_rate = dropout
        self.input_dim = input_dim
        self.fc1 = nn.Linear(1, num_of_nodes)
        self.fc2 = nn.Linear(num_of_nodes, num_of_nodes)
        self.fc3 = nn.Linear(2 * input_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLPDecoder.

        Args:
            z (torch.Tensor): Latent node representations of shape [batch, input_dim].

        Returns:
            torch.Tensor: Reconstructed adjacency matrix of shape [batch * num_of_nodes, batch * num_of_nodes].
        """
        z = F.leaky_relu(self.fc1(z.unsqueeze(-1)))
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        z = F.leaky_relu(self.fc2(z))
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        z = z.permute(0, 2, 1).reshape(-1, self.input_dim)

        z_pairs = torch.cat([z[i].unsqueeze(0).repeat(z.size(0), 1, 1) for i in range(z.size(0))], dim=1)
        z_pairs = torch.cat((z_pairs, z_pairs.transpose(0, 1)), dim=-1)

        # x = F.leaky_relu(self.bn1(self.fc2(z_pairs)))
        x = F.leaky_relu(self.fc3(z_pairs))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = torch.sigmoid(self.fc4(x))

        adj_pred = x.view(z.size(0), z.size(0))
        return adj_pred


class MLPDecoder(nn.Module):
    """
    MLP-based decoder for reconstructing the adjacency matrix in a Graph Variational Autoencoder.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.5):
        """
        Initializes the MLPDecoder with specified dimensions and dropout rate.

        Args:
            input_dim (int): Dimension of the input latent node representations.
            hidden_dim (int): Dimension of the hidden layer in the decoder.
            dropout (float): Dropout rate to use after the activation function.
        """
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(2 * input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout_rate = dropout

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLPDecoder.

        Args:
            z (torch.Tensor): Latent node representations of shape [num_nodes, input_dim].

        Returns:
            torch.Tensor: Reconstructed adjacency matrix of shape [num_nodes, num_nodes].
        """
        z_pairs = torch.cat([z[i].unsqueeze(0).repeat(z.size(0), 1, 1) for i in range(z.size(0))], dim=1)
        z_pairs = torch.cat((z_pairs, z_pairs.transpose(0, 1)), dim=-1)

        # x = F.leaky_relu(self.bn1(self.fc1(z_pairs)))
        x = F.leaky_relu(self.fc1(z_pairs))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = torch.sigmoid(self.fc2(x))

        adj_pred = x.view(z.size(0), z.size(0))
        return adj_pred


class GVAE(nn.Module):
    """
    Graph Variational Autoencoder for generating graph structures.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 z_dim: int,
                 dropout: float = 0.0,
                 node_pooling: bool = False,
                 decode_from_pooled: bool = False
                 ):
        """
        Initialize the GVAE.
        :param input_dim: Dimension of input features per node.
        :param hidden_dim: Dimension of hidden layers.
        :param z_dim: Dimension of the latent space (will be split into mu and logvar).
        """
        super(GVAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = GCNGraphEncoder(input_dim, hidden_dim, z_dim, dropout, global_pooling=node_pooling)

        if decode_from_pooled:
            self.adj_decoder = MLPPooledDecoder(z_dim // 2, hidden_dim, dropout, num_of_nodes=30)
            self.node_feature_decoder = NodeFeaturePooledDecoder(z_dim // 2, hidden_dim, input_dim, dropout, num_of_nodes=30)
        else:
            self.adj_decoder = MLPDecoder(z_dim // 2, hidden_dim, dropout)
            self.node_feature_decoder = NodeFeatureDecoder(z_dim // 2, hidden_dim, input_dim, dropout=dropout)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """
        Initialize the weights of the model.
        :param m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: Mean from the encoder's latent space.
        :param logvar: Log variance from the encoder's latent space.
        :return: Sampled z following the distributions.
        """
        if self.training:
            std = torch.exp(logvar / 2)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, data: Data, training: bool = True):
        """
        Forward pass of the GVAE.
        :param data: Graph data containing node features and edge indices.
        :return: A tuple of reconstructed node features, mu, and logvar.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        z = self.encoder(x, edge_index, batch)

        mu, logvar = z.chunk(2, dim=-1)  # Split z into mu and logvar
        if training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        adj_pred = self.adj_decoder(z)
        node_ft_pred = self.node_feature_decoder(z)

        return adj_pred, node_ft_pred, mu, logvar


def gvae_loss_function(
        recon_x_tensor: torch.Tensor,
        recon_adj_tensor: torch.Tensor,
        batched_data: Batch,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        class_bce_weights = None,
        adj_weight: float = 1.0,
        adj_pos_weight: float = 10.0,
        adj_neg_weight: float = 0.1,
        kl_weight: float = 1.0):
    """
    Compute the Graph Variational Autoencoder loss function.

    :param recon_x_tensor: Reconstructed graph/node features [batch_size, num_features]
    :param recon_adj_tensor: Reconstructed adjacency matrix
    :param batched_data: Ground truth torch_geometrics batched data
    :param mu: Mean of the latent space distribution [batch_size, latent_dim]
    :param logvar: Log variance of the latent space distribution [batch_size, latent_dim]
    :param class_bce_weights:
    :param adj_weight: Weight for the adjacency matrix reconstruction loss
    :param adj_pos_weight:
    :param adj_neg_weight:
    :param kl_weight: Weight for the KL divergence loss
    :return: Total loss, reconstruction loss for node features, reconstruction loss for adjacency matrix, KL divergence
    """
    # Node feature reconstruction loss
    recon_loss_ft_class = F.binary_cross_entropy(recon_x_tensor[:, :-4], batched_data.x[:, :-4],
                                                 weight=class_bce_weights,
                                                 reduction='sum')
    recon_loss_ft_cont = F.mse_loss(recon_x_tensor[:, -4:], batched_data.x[:, -4:], reduction='sum')
    recon_loss_ft = recon_loss_ft_class + recon_loss_ft_cont

    # Adjacency matrix reconstruction loss
    gt_adj = to_dense_adj(batched_data.edge_index, max_num_nodes=batched_data.num_nodes)[0]
    recon_loss_adj = weighted_binary_cross_entropy(recon_adj_tensor, gt_adj,
                                                   positive_weight=adj_pos_weight,
                                                   negative_weight=adj_neg_weight,
                                                   reduction='sum')

    # recon_loss_adj += dice_loss(recon_adj_tensor, adj)
    # recon_loss_adj = focal_loss(recon_adj_tensor, adj, alpha=0.25, gamma=2.0)
    # recon_loss_adj = tversky_loss(recon_adj_tensor, adj, alpha=0.25, beta=0.25)

    # KL divergence (regularization term) # TODO: mean vs sum?
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss_ft + adj_weight * recon_loss_adj + kl_weight * kl_div

    return total_loss, recon_loss_ft, recon_loss_ft_class, recon_loss_ft_cont, recon_loss_adj, kl_div
