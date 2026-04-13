import gc
import argparse
import datetime
import os
import logging
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm.autonotebook import tqdm
from omegaconf import OmegaConf
from swomo.data.dataset import SurgicalDataset
from swomo.graph_encoder.graph_segclip_masked import *
from swomo.utils.get_scene_graph import GraphConstructor

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def create_logger(log_file_path):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG) 

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def build_loaders(video_folder, mode, dataset_name, class_size, ignore_index, size, num_graph_in_3d, overlap_size, augmentation, batch_size, num_workers):
    dataset = SurgicalDataset(
        video_folder=video_folder,
        split_mode=mode,
        dataset_name=dataset_name,
        class_size=class_size,
        ignore_index=ignore_index,
        sample_size=size, 
        sample_n_frames=num_graph_in_3d,
        overlap_size=overlap_size,
        apply_augmentation=augmentation if mode == "train" else False,
        train_graph_encoder=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, device, graphconstr=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if not k.endswith("path")}

        # Outside the dataset, bcs need cuda for optical flow info for graph construction
        if train_loader.dataset.apply_augmentation:
            graphs = []
            for n_b in range(batch['image'].shape[0]): 
                graph = graphconstr.create_scene_graph(batch['graph_cond'][n_b], batch['graph_segmentation'][n_b])
                graphs.append(graph)
            graphs = Batch.from_data_list(graphs)
            batch["graph"] = graphs.to(device)

        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

def valid_epoch(model, valid_loader, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if not k.endswith("path")}
        loss = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def train(name,
          dataset_name="cataract",
          class_size=18,
          ignore_index="first",
          num_graph_in_3d=16,
          overlap_size=1, 
        
          size=128,
          batch_size=32,
          num_workers=8,
          weight_decay=1e-5,
          patience=1,
          factor=0.8,
          epochs=200,
      
          graph_encoder_lr=1e-5,
          graph_input_dim=21,
          graph_hidden_dim=256,
          graph_embedding_dim=256,
          graph_conv_type="GCNConv",
          graph_norm_type="GroupNorm",
          graph_encoder_ckpt=None,

          image_encoder_lr=1e-6,
          image_embedding_dim=256,
          image_encoder_config="",
          image_encoder_ckpt="",
          
          segmentation_encoder_lr=1e-7,
          segmentation_embedding_dim=256,
          segmentation_encoder_config="",
          segmentation_encoder_ckpt="",

          augmentation=False,
          trainable=True,
          temperature=1.0,
          dropout=0.25,
          data_root="",
          log_dir="",
          device="cuda",
          **kwargs):

    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if slurm_job_id is not None:
        exp_dir = f"{log_dir}/graphencoder_{name}_{dataset_name}_{slurm_job_id}-{timestamp}"
    else:
        exp_dir = f"{log_dir}/graphencoder_{name}_{dataset_name}-{timestamp}"

    checkpoint_dir = f"{exp_dir}/checkpoints"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))

    log = create_logger(f"{exp_dir}/logfile.log")
    log.info(f"{torch.cuda.is_available()=}")

    train_loader = build_loaders(video_folder=data_root, mode="train", dataset_name=dataset_name, class_size=class_size, ignore_index=ignore_index, size=size, num_graph_in_3d=num_graph_in_3d, overlap_size=overlap_size, augmentation=augmentation, batch_size=batch_size, num_workers=num_workers)
    valid_loader = build_loaders(video_folder=data_root, mode="val", dataset_name=dataset_name, class_size=class_size, ignore_index=ignore_index, size=size, num_graph_in_3d=num_graph_in_3d, overlap_size=overlap_size, augmentation=augmentation, batch_size=batch_size, num_workers=num_workers)

    if name == "segclip":
        graph_encoder = GraphEncoder(graph_input_dim, graph_hidden_dim, graph_embedding_dim, trainable, device, dropout, graph_conv_type, graph_norm_type, graph_encoder_ckpt)
        segmentation_encoder = SegmentationEncoder(device, segmentation_encoder_config, segmentation_encoder_ckpt)
        model = SegClipModel(temperature, segmentation_embedding_dim, segmentation_encoder, graph_encoder).to(device)
        params = [{"params": model.graph_encoder.parameters(), "lr": graph_encoder_lr},
                 # {"params": model.segmentation_encoder.parameters(), "lr": segmentation_encoder_lr}
                 ]
        
    elif name == "masked":
        graph_encoder = GraphEncoder(graph_input_dim, graph_hidden_dim, graph_embedding_dim, trainable, device, dropout, graph_conv_type, graph_norm_type, graph_encoder_ckpt)
        image_encoder = ImageEncoder(device, image_encoder_config, image_encoder_ckpt)
        model = MaskedLocalModel(ignore_index, dropout, image_embedding_dim, image_encoder, graph_embedding_dim, graph_encoder).to(device)
        params = [{"params": model.basic_transformer.parameters(), "lr": image_encoder_lr},
                  {"params": model.graph_encoder.parameters(), "lr": graph_encoder_lr}
                 ]

    #TODO: add beta, eps
    optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=factor)

    if augmentation:
        log.info("### Using data augmentation ###")
        log.info("### Graphs will be formed on the fly and runs much slower ###")
        graphconstr = GraphConstructor(num_graph_in_3d=num_graph_in_3d, device=device, num_classes=class_size, background_label=kwargs['background_label'], anatomy_label=kwargs['anatomy_label'], tool_label=kwargs['tool_label'])

    best_loss = float('inf')
    for epoch in range(epochs):
        log.info(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, device, graphconstr if augmentation else None)
        
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader, device)

        log.info(f"### Training Loss: {train_loss} ###")
        log.info(f"### Validation Loss: {valid_loss} ###")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'latest_val_loss.pth'))

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            log.info("### New best validation loss ###")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_val_loss.pth'))

        lr_scheduler.step(valid_loss.avg)

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, choices=["segclip", "masked"])
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()   

    config = OmegaConf.load(args.config)
    train(name=args.name, **config)