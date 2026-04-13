import yaml
import random
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from swomo.taming.taming.models.vqgan import VQModel
from swomo.graph_encoder.graph_vae import GCNGraphEncoder
from swomo.graph_encoder.mm_transformer_module import BasicTransformerBlock

def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    return (x - mean) / std

def cross_entropy(preds, targets):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    return loss

def mask_to_bbox(binary_mask):
    bboxs = []
    for mask in binary_mask:
        non_zero_coords = torch.nonzero(mask, as_tuple=False)
        # So not every frame in sequence has the chosen object to mask. So need to handle this
        if non_zero_coords.numel() == 0:
            ymin, xmin, ymax, xmax = None, None, None, None
            bbox = (xmin, ymin, xmax, ymax)
        else:        
            ymin, xmin = torch.min(non_zero_coords, dim=0).values
            ymax, xmax = torch.max(non_zero_coords, dim=0).values
            bbox = (xmin.item(), ymin.item(), xmax.item(), ymax.item())
        
        bboxs.append(bbox)
    return bboxs

def random_object_masked_image(batch, ignore_index):
    # Selecting random existing object in segmentation map to create masked image
    bbox_masks = []
    B, S, D, H, W = batch["image"].size()
    class_labels = batch["class_label"]
    
    if ignore_index in ["first", "last"]:
        ignore_idx = 0 if ignore_index == "first" else -1
    else: raise ValueError(f"Invalid ignore_index: {ignore_index}")

    kernel = 21
    padding = kernel // 2
    conv_kernel = torch.ones((1, 1, kernel, kernel), device=class_labels.device)

    for i in range(B):
        # If graph is empty, then mask the ignore/background class.  
        # Else do not mask ignore/background class. So manually set ignore_idx to zero
        if ignore_idx == 0: 
            only_ignore_class = (class_labels[i][ignore_idx] == 1) and (class_labels[i][1:].sum() == 0)
        else:
            only_ignore_class = (class_labels[i][ignore_idx] == 1) and (class_labels[i][:-1].sum() == 0)
        if not only_ignore_class:
            class_labels[i][ignore_idx] = 0
            
        non_zero_indices = torch.nonzero(class_labels[i], as_tuple=False).squeeze(1)
        random_class = torch.randint(0, len(non_zero_indices), (1,)).item()
        random_class = non_zero_indices[random_class]

        # Dilate the mask (Dilate primary knive even more)
        mask = batch["segmentation"][i][:,random_class,:,:]
        mask = F.conv2d(mask.unsqueeze(1), conv_kernel, padding=padding)
        mask = (mask > 0).float().squeeze(1)
        
        bbox = mask_to_bbox(mask)
        bbox_mask = torch.ones(S, D, H, W, dtype=mask.dtype, device=mask.device)
        
        for ix, bbx in enumerate(bbox):

            bbox_mask[ix, :, bbx[1]:bbx[3], bbx[0]:bbx[2]] = 0
        bbox_masks.append(bbox_mask)

    bbox_masks = torch.stack(bbox_masks)
    masked_image = batch["image"] * bbox_masks
    return masked_image
    

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


class ImageEncoder(nn.Module):
    """
    Encode image to a fixed size vector
    """
    def __init__(
        self,
        device,
        image_encoder_config,
        image_encoder_ckpt
    ):
        super().__init__()
        self.config = self.load_config(image_encoder_config)
        self.model = self.load_vqgan(self.config, image_encoder_ckpt).to(device)
        
    def forward(self, x):
        sequence_dim = x.size(1)
        x = x.view(-1, *x.shape[2:])
        z, _, [_, _, indices] = self.model.encode(x)
        encodings = z.squeeze(1).view(z.size(0), -1)
        encodings = encodings.view(-1, sequence_dim, encodings.size(-1))
        # encodings = F.normalize(encodings, p=2, dim=-1)
        return encodings
    
    def load_config(self, config_path, display=False):
        config = OmegaConf.load(config_path)
        if display:
            print(yaml.dump(OmegaConf.to_container(config)))
        return config

    def load_vqgan(self, config, ckpt_path=None):
        model = VQModel(**config.model.params)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        if sd is not None:  
            missing, unexpected = model.load_state_dict(sd, strict=False)
        return model.eval()


class SegmentationEncoder(nn.Module):
    """
    Encode segmentation to a fixed size vector
    """
    def __init__(
        self,
        device,
        segmentation_encoder_config,
        segmentation_encoder_ckpt
    ):
        super().__init__()
        self.config = self.load_config(segmentation_encoder_config)
        self.model = self.load_vqgan(self.config, segmentation_encoder_ckpt).to(device)
        
    def forward(self, x):
        sequence_dim = x.size(1)
        x = x.view(-1, *x.shape[2:])
        z, _, [_, _, indices] = self.model.encode(x)
        encodings = z.squeeze(1).view(z.size(0), -1)
        encodings = encodings.view(-1, sequence_dim, encodings.size(-1))
        # encodings = F.normalize(encodings, p=2, dim=-1)
        return encodings

    def load_config(self, config_path, display=False):
        config = OmegaConf.load(config_path)
        if display:
            print(yaml.dump(OmegaConf.to_container(config)))
        return config

    def load_vqgan(self, config, ckpt_path=None):
        model = VQModel(**config.model.params)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        if sd is not None:  
            missing, unexpected = model.load_state_dict(sd, strict=False)
        return model.eval()


class GraphEncoder(nn.Module):
    """
    Encode graphs to a fixed size vector
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        z_dim,
        trainable,
        device="cpu",
        dropout=0.5,
        graph_conv_type="GCNConv",
        graph_norm_type="GroupNorm",
        graph_encoder_ckpt=None
    ):
        super().__init__()
        self.model = GCNGraphEncoder(input_dim, hidden_dim, z_dim, dropout, graph_conv_type, graph_norm_type, global_pooling=True)
        if graph_encoder_ckpt is not None:
            graph_encoder_weights = torch.load(graph_encoder_ckpt, map_location=device)
            graph_encoder_weights = {k.replace("graph_encoder.model.", ""): v for k, v in graph_encoder_weights.items() if k.startswith("graph_encoder")}
            self.model.load_state_dict(graph_encoder_weights)
            self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, graph):
        # graph.x = normalize(graph.x, 0.0, 1.0).to(torch.float32)
        # graph.x[:, -4:] = (graph.x[:, -4:] + 1.0) / 2.0
        graph.x = graph.x.to(torch.float32)
        encodings = self.model(graph.x, graph.edge_index, graph.batch)
        batch_size, sequence_dim = graph.batch_size, int(encodings.size(0)/graph.batch_size)
        encodings = encodings.view(batch_size, sequence_dim, encodings.size(-1))
        # encodings = F.normalize(encodings, p=2, dim=-1)
        return encodings
    

class SegClipModel(nn.Module):
    def __init__(
        self,
        temperature,
        segmentation_dim,
        segmentation_encoder,
        graph_encoder
    ):
        super().__init__()
        self.temperature = temperature
        self.segmentation_dim = segmentation_dim
        self.segmentation_encoder = segmentation_encoder
        self.graph_encoder = graph_encoder

    def forward(self, batch):
        graph_embeddings = self.graph_encoder(batch["graph"])
        graph_embeddings = graph_embeddings.view(graph_embeddings.size(0), -1)
        segmentation_embeddings = self.segmentation_encoder(batch["segmentation"])
        segmentation_embeddings = segmentation_embeddings.view(segmentation_embeddings.size(0), -1)
        
        # Calculating the Loss
        logits = (graph_embeddings @ segmentation_embeddings.T) / self.temperature
        segmentation_similarity = segmentation_embeddings @ segmentation_embeddings.T
        # graphs_similarity = graph_embeddings @ graph_embeddings.T
        
        targets = F.softmax(
            (segmentation_similarity) / self.temperature, dim=-1
        )
        graphs_loss = cross_entropy(logits, targets)
        segmentation_loss = cross_entropy(logits.T, targets.T)
        loss =  (segmentation_loss + graphs_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

class MaskedLocalModel(nn.Module):
    def __init__(
        self,
        ignore_index,
        dropout,
        image_embedding_dim,
        image_encoder,
        graph_embedding_dim,
        graph_encoder
    ):
        super(MaskedLocalModel, self).__init__()
        self.ignore_index = ignore_index
        self.image_encoder = image_encoder
        self.graph_encoder = graph_encoder
        #TODO: Experiment more with n_heads and d_head size.
        self.basic_transformer = BasicTransformerBlock(dim=image_embedding_dim, n_heads=16, d_head=64, dropout=dropout, context_dim=graph_embedding_dim)
        self.criterion = nn.MSELoss()
    
    def forward(self, batch):
        masked_image = random_object_masked_image(batch, self.ignore_index)
        
        graph_embeddings = self.graph_encoder(batch["graph"])
        image_gt_embeddings = self.image_encoder(batch["image"]).contiguous()
        image_masked_embeddings = self.image_encoder(masked_image).contiguous()
        image_reconstructed_embeddings = self.basic_transformer(image_masked_embeddings, context=graph_embeddings)
        
        # Calculating the Loss
        loss = self.criterion(image_reconstructed_embeddings, image_gt_embeddings)
        return loss
