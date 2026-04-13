import os
import math
import wandb
import random
import time
import logging
import inspect
import argparse
import datetime
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import EMAModel

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from swomo.data.dataset import SurgicalDataset
from swomo.models.videoldm_unet import VideoLDMUNet3DConditionModel
from swomo.models.videoldm_controlnet import ControlNetModel
from swomo.graph_encoder.graph_segclip_masked import GraphEncoder
from swomo.pipelines.pipeline_conditional_animation import ConditionalAnimationPipeline
from swomo.pipelines.pipeline_controlnet_animation import ControlNetAnimationPipeline
from swomo.utils.util import save_videos_grid, save_videos_grid_from_paths
from swomo.utils.get_scene_graph import GraphConstructor

logger = get_logger(__name__, log_level="INFO")

def main(
    name: str,
    use_wandb: bool,

    is_image: bool,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,

    cfg_random_null_text_ratio: float = 0.1,
    cfg_random_null_img_ratio: float = 0.0,
    
    finetuned_unet_path: Optional[str] = None,
    finetuned_autoencoder_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    only_save_latest_checkpoint: bool = False,

    unet_additional_kwargs: Dict = {},
    unet_graph_kwargs: Dict = {},
    use_ema: bool = False,
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision: Optional[str] = "fp16",
    enable_xformers_memory_efficient_attention: bool = True,

    seed: Optional[int] = 42,
    is_debug: bool = False,
):
    check_min_version("0.10.0.dev0")
    *_, config = inspect.getargvalues(inspect.currentframe())
    config = {k: v for k, v in config.items() if k != 'config' and k != '_'}

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True if not is_image else False)
    init_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=3600))

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )
    
    if seed is not None:
        set_seed(seed)

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process and (not is_debug) and use_wandb:
        project_name = "text_image_to_video" if not is_image else "image_finetune"
        wandb.init(project=project_name, name=folder_name, config=config)
    accelerator.wait_for_everyone()

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    vae             = AutoencoderKL.from_pretrained(finetuned_autoencoder_path if finetuned_autoencoder_path else pretrained_model_path, subfolder="vae")
    tokenizer       = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder    = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    unet            = VideoLDMUNet3DConditionModel.from_pretrained(
        finetuned_unet_path if finetuned_unet_path else pretrained_model_path,
        subfolder="unet",
        variant=unet_additional_kwargs['variant'],
        use_temporal=True if not is_image else False,
        temp_pos_embedding=unet_additional_kwargs['temp_pos_embedding'],
        augment_temporal_attention=unet_additional_kwargs['augment_temporal_attention'],
        n_frames=train_data.sample_n_frames if not is_image else 2,
        n_temp_heads=unet_additional_kwargs['n_temp_heads'],
        first_frame_condition_mode=unet_additional_kwargs['first_frame_condition_mode'],
        use_frame_stride_condition=unet_additional_kwargs['use_frame_stride_condition'],
        
        class_embeddings_concat=unet_graph_kwargs.get('class_embeddings_concat', False),
        class_embed_type=unet_graph_kwargs.get('class_embed_type', None),
        time_embedding_dim=unet_graph_kwargs.get('time_embedding_dim', None),            
        ignore_mismatched_sizes=unet_graph_kwargs.get('ignore_mismatched_sizes', False),
        use_safetensors=True
    )
    if train_data.return_cond_videos:
        controlnet = ControlNetModel.from_unet(unet)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    if not train_data.return_cond_videos:
        unet.train()
    else:
        controlnet.train()
        unet.requires_grad_(False)

    if use_ema:
        if not train_data.return_cond_videos:
            ema_unet = VideoLDMUNet3DConditionModel.from_pretrained(
                pretrained_model_path,
                subfolder="unet",
                variant=unet_additional_kwargs['variant'],
                use_temporal=True if not is_image else False,
                temp_pos_embedding=unet_additional_kwargs['temp_pos_embedding'],
                augment_temporal_attention=unet_additional_kwargs['augment_temporal_attention'],
                n_frames=train_data.sample_n_frames if not is_image else 2,
                n_temp_heads=unet_additional_kwargs['n_temp_heads'],
                first_frame_condition_mode=unet_additional_kwargs['first_frame_condition_mode'],
                use_frame_stride_condition=unet_additional_kwargs['use_frame_stride_condition'],
                
                class_embeddings_concat=unet_graph_kwargs.get('class_embeddings_concat', False),
                class_embed_type=unet_graph_kwargs.get('class_embed_type', None),
                time_embedding_dim=unet_graph_kwargs.get('time_embedding_dim', None),            
                ignore_mismatched_sizes=unet_graph_kwargs.get('ignore_mismatched_sizes', False),
                use_safetensors=True
            )
            ema_unet = EMAModel(ema_unet.parameters(), decay=ema_decay, model_cls=VideoLDMUNet3DConditionModel, model_config=ema_unet.config)
        else:
            ema_controlnet = ControlNetModel.from_unet(unet)
            ema_controlnet = EMAModel(ema_controlnet.parameters(), decay=ema_decay, model_cls=ControlNetModel, model_config=ema_controlnet.config)

    #TODO: Kinda messy way to do this with "if not train_data.return_cond_videos"
    # Set unet trainable parameters
    # For controlnet training, unet is frozen and all controlnet parameters are trainable
    if not train_data.return_cond_videos:
        train_all_parameters = False
        for trainable_module_name in trainable_modules:
            if trainable_module_name == 'all':
                unet.requires_grad_(True)
                train_all_parameters = True
                break

        if not train_all_parameters:
            unet.requires_grad_(False)
            for name, param in unet.named_parameters():
                for trainable_module_name in trainable_modules:
                    if trainable_module_name in name:
                        param.requires_grad = True
                        break

    # Enable xformers
    if enable_xformers_memory_efficient_attention and int(torch.__version__.split(".")[0]) < 2:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            if train_data.return_cond_videos:
                controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if use_ema:
                if not train_data.return_cond_videos:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                else:
                    ema_controlnet.save_pretrained(os.path.join(output_dir, "controlnet_ema"))
                    
            for i, model in enumerate(models):
                if not train_data.return_cond_videos:
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                else:
                    model.save_pretrained(os.path.join(output_dir, "controlnet"))    
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        if use_ema:
            if not train_data.return_cond_videos:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), VideoLDMUNet3DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
            else:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "controlnet_ema"), ControlNetModel)
                ema_controlnet.load_state_dict(load_model.state_dict())
                ema_controlnet.to(accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if not train_data.return_cond_videos:
                load_model = VideoLDMUNet3DConditionModel.from_pretrained(input_dir, subfolder="unet")
            else:
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable gradient checkpointing
    if gradient_checkpointing:
        if not train_data.return_cond_videos:
            unet.enable_gradient_checkpointing()
        else:
            controlnet.enable_gradient_checkpointing()
            
    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)
    
    if not train_data.return_cond_videos:
        trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    else:
        trainable_params = list(filter(lambda p: p.requires_grad, controlnet.parameters()))
        
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    logger.info(f"trainable params number: {len(trainable_params)}")
    logger.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Get the training dataset
    if train_data['dataset'] == "surgical":
        if hasattr(validation_data, "path_to_first_frames") and validation_data.path_to_first_frames is not None:
            train_data.validation_first_frames = validation_data.path_to_first_frames
        if hasattr(validation_data, "path_to_cond_videos") and validation_data.path_to_cond_videos is not None:
            train_data.validation_cond_videos = validation_data.path_to_cond_videos
        
        # Will return None if non-graph conditioned and return_graph_emb will catch it
        # Passing validation graph and paths to SurgicalDataset and will return the embedding and list of paths of the video
        train_data.validation_graphs = validation_data.path_to_graphs
        train_dataset = SurgicalDataset(**train_data)
        validation_seq_paths = train_dataset.validation_seq_paths
        if train_data.return_cond_videos:
            validation_cond_seq_paths = train_dataset.validation_cond_seq_paths
        
    else:
        raise ValueError(f"Unknown dataset {train_data['dataset']}")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not train_data.return_cond_videos:
        validation_pipeline = ConditionalAnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        )
    else:
        validation_pipeline = ControlNetAnimationPipeline(
            unet=unet, controlnet=controlnet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        )

    validation_pipeline.enable_vae_slicing()

    # Prepare everything with our `accelerator`.
    if not train_data.return_cond_videos:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if use_ema:
        if not train_data.return_cond_videos:
            ema_unet.to(accelerator.device)
        else:
            ema_controlnet.to(accelerator.device)
    
    # Move text_encode and vae to gpu and cast to weight_dtype
    if train_data.return_cond_videos:
        #TODO: This is breaking with bf16 mixed precision, which I originally used to train this model
        unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if train_data.return_graph_emb:
        embedding_type = train_data.embedding_type
        if 'model_masked' in train_data:
            m_graph_emb_masked = GraphEncoder(train_data.graph_input_dim, train_data.graph_hidden_dim, 
                                              train_data.graph_embedding_dim, train_data.trainable,
                                              graph_conv_type = train_data.graph_conv_type, 
                                              graph_norm_type = train_data.graph_norm_type, 
                                              graph_encoder_ckpt = train_data.model_masked)
            m_graph_emb_masked.to(accelerator.device)
            m_graph_emb_masked.eval()
        if 'model_segclip' in train_data:
            m_graph_emb_segclip = GraphEncoder(train_data.graph_input_dim, train_data.graph_hidden_dim, 
                                               train_data.graph_embedding_dim, train_data.trainable, 
                                               graph_conv_type = train_data.graph_conv_type, 
                                               graph_norm_type = train_data.graph_norm_type, 
                                               graph_encoder_ckpt = train_data.model_segclip)
            m_graph_emb_segclip.to(accelerator.device)
            m_graph_emb_segclip.eval()

        def get_embedding(scene_graph, embedding_type):
            if embedding_type == 'masked':
                graph_embeddings = m_graph_emb_masked(scene_graph)
            elif embedding_type == 'segclip': 
                graph_embeddings = m_graph_emb_segclip(scene_graph)
            elif embedding_type == 'combined':
                graph_embeddings_masked = m_graph_emb_masked(scene_graph)
                graph_embeddings_segclip = m_graph_emb_segclip(scene_graph)
                graph_embeddings = torch.cat((graph_embeddings_masked, graph_embeddings_segclip), dim=-1)
            return graph_embeddings

    if train_data.apply_augmentation and train_data.return_graph_emb:
        logger.info("Using data augmentation")
        logger.info("Graphs will be formed on the fly")
        graphconstr = GraphConstructor(num_graph_in_3d=train_data.sample_n_frames, device=accelerator.device, num_classes=train_data.class_size, background_label=train_data.background_label, anatomy_label=train_data.anatomy_label, tool_label=train_data.tool_label)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Save validation's first frame and gt sequence
    height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
    width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

    if hasattr(validation_data, "path_to_first_frames") and validation_data.path_to_first_frames is not None:
        save_path_gt_seq = f"{output_dir}/samples/gt/gt_seq.gif"
        save_videos_grid_from_paths(validation_seq_paths, save_path_gt_seq, resize=(height, width))
        save_path_first_frame = f"{output_dir}/samples/gt/first_frame.gif"
        save_videos_grid_from_paths([[first_frame] for first_frame in validation_data.path_to_first_frames], save_path_first_frame, resize=(height, width)) 
    if hasattr(validation_data, "path_to_cond_videos") and validation_data.path_to_cond_videos is not None:
        save_path_cond_seq = f"{output_dir}/samples/gt/cond_seq.gif"
        save_videos_grid_from_paths(validation_cond_seq_paths, save_path_cond_seq, resize=(height, width))

    # Load pretrained unet weights
    if resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        accelerator.load_state(resume_from_checkpoint)
        global_step = int(resume_from_checkpoint.split("-")[-1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
        logger.info(f"global_step: {global_step}")
        logger.info(f"first_epoch: {first_epoch}")
    else:
        initial_global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(0, max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_main_process)

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        train_grad_norm = 0.0
        data_loading_time = 0.0
        prepare_everything_time = 0.0
        network_forward_time = 0.0
        network_backward_time = 0.0

        t0 = time.time()
        for step, batch in enumerate(train_dataloader):
            t1 = time.time()
            if cfg_random_null_text_ratio > 0.0:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
            
            # Data batch sanity check
            if accelerator.is_main_process and epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                if train_data.return_cond_videos:
                    cond_video_values = batch['cond_video_values'].cpu()
                    cond_video_values = rearrange(cond_video_values, "b f c h w -> b c f h w") 

                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'no_text-{idx}'}.gif", rescale=True)
                
            ### >>>> Training >>>> ###
            with accelerator.accumulate(unet if not train_data.return_cond_videos else controlnet):
                # Convert videos to latent space            
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist
                latents = latents.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * vae.config.scaling_factor

                if train_data.return_cond_videos:
                    cond_video_values = batch['cond_video_values'].to(weight_dtype)
                    cond_video_values = rearrange(cond_video_values, "b f c h w -> b c f h w") 

                if unet_additional_kwargs["first_frame_condition_mode"] != "none":
                    # Get first frame latents
                    first_frame_latents = latents[:, :, 0:1, :, :]

                # Sample noise that we'll add to the latents
                if unet_additional_kwargs['noise_sampling_method'] == 'vanilla':
                    noise = torch.randn_like(latents)
                elif unet_additional_kwargs['noise_sampling_method'] == 'pyoco_mixed':
                    noise_alpha_squared = float(unet_additional_kwargs['noise_alpha']) ** 2
                    shared_noise = torch.randn_like(latents[:, :, 0:1, :, :]) * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared))
                    ind_noise = torch.randn_like(latents) * math.sqrt(1 / (1 + noise_alpha_squared))
                    noise = shared_noise + ind_noise
                elif unet_additional_kwargs['noise_sampling_method'] == 'pyoco_progressive':
                    noise_alpha_squared = float(unet_additional_kwargs['noise_alpha']) ** 2
                    noise = torch.randn_like(latents)
                    ind_noise = torch.randn_like(latents) * math.sqrt(1 / (1 + noise_alpha_squared))
                    for i in range(1, noise.shape[2]):
                        noise[:, :, i, :, :] = noise[:, :, i - 1, :, :] * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared)) + ind_noise[:, :, i, :, :]
                else:
                    raise ValueError(f"Unknown noise sampling method {unet_additional_kwargs['noise_sampling_method']}")

                bsz = latents.shape[0]
            
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
            
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if cfg_random_null_img_ratio > 0.0:
                    for i in range(first_frame_latents.shape[0]):
                        if random.random() <= cfg_random_null_img_ratio:
                            first_frame_latents[i, :, :, :, :] = noisy_latents[i, :, 0:1, :, :]

                # Remove the first noisy latent from the latents if we're conditioning on the first frame
                if unet_additional_kwargs["first_frame_condition_mode"] != "none":
                    noisy_latents = noisy_latents[:, :, 1:, :, :]
            
                # Get the text embedding for conditioning
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                timesteps = repeat(timesteps, "b -> b f", f=video_length)
                timesteps = rearrange(timesteps, "b f -> (b f)")

                frame_stride = None
                if unet_additional_kwargs["use_frame_stride_condition"]:
                    frame_stride = batch['stride'].to(latents.device)
                    frame_stride = frame_stride.long()
                    frame_stride = repeat(frame_stride, "b -> b f", f=video_length)
                    frame_stride = rearrange(frame_stride, "b f -> (b f)")

                class_label_condition_mode = unet_graph_kwargs.get('class_label_condition_mode', False)
                if class_label_condition_mode:
                    if train_data.apply_augmentation:
                        graphs = []
                        for n_b in range(train_batch_size): 
                            graph = graphconstr.create_scene_graph(batch['graph_cond'][n_b], batch['graph_segmentation'][n_b])
                            graphs.append(graph)
                        graphs = Batch.from_data_list(graphs)
                        batch["graph"] = graphs.to(latents.device)
                    
                    #TODO: This is batching a batch, and might not be the same anymore
                    class_labels = get_embedding(batch["graph"], embedding_type=embedding_type)
                    class_labels = class_labels.to(latents.device)

                t2 = time.time()
                down_block_additional_residuals = None
                mid_block_additional_residual = None
                
                if train_data.return_cond_videos:
                    controlnet_kwargs = dict(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=cond_video_values,
                        return_dict=False,
                    )
                    if unet_additional_kwargs["first_frame_condition_mode"] != "none":
                        controlnet_kwargs["first_frame_latents"] = first_frame_latents
                        controlnet_kwargs["frame_stride"] = frame_stride

                    if class_label_condition_mode:
                        controlnet_kwargs["class_labels"] = class_labels
                    
                    down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_kwargs)
                    down_block_additional_residuals = [sample.to(dtype=weight_dtype) for sample in down_block_res_samples]
                    mid_block_additional_residual = mid_block_res_sample.to(dtype=weight_dtype)

                # Predict the noise residual and compute loss
                if unet_additional_kwargs["first_frame_condition_mode"] != "none" and not class_label_condition_mode:
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, first_frame_latents=first_frame_latents, frame_stride=frame_stride, down_block_additional_residuals=down_block_additional_residuals, mid_block_additional_residual=mid_block_additional_residual).sample
                    loss = F.mse_loss(model_pred.float(), target.float()[:, :, 1:, :, :], reduction="mean")
                elif unet_additional_kwargs["first_frame_condition_mode"] != "none" and class_label_condition_mode:
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, first_frame_latents=first_frame_latents, frame_stride=frame_stride, class_labels=class_labels, down_block_additional_residuals=down_block_additional_residuals, mid_block_additional_residual=mid_block_additional_residual).sample
                    loss = F.mse_loss(model_pred.float(), target.float()[:, :, 1:, :, :], reduction="mean")            
                elif unet_additional_kwargs["first_frame_condition_mode"] == "none" and not class_label_condition_mode:
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, down_block_additional_residuals=down_block_additional_residuals, mid_block_additional_residual=mid_block_additional_residual).sample
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif unet_additional_kwargs["first_frame_condition_mode"] == "none" and class_label_condition_mode:
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, class_labels=class_labels, down_block_additional_residuals=down_block_additional_residuals, mid_block_additional_residual=mid_block_additional_residual).sample
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    raise ValueError("Unexpected combination of 'first_frame_condition_mode' and 'class_label_condition_mode'")

                t3 = time.time()
                
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(unet.parameters() if not train_data.return_cond_videos else controlnet.parameters(), max_grad_norm)
                    avg_grad_norm = accelerator.gather(grad_norm.repeat(train_batch_size)).mean()
                    train_grad_norm += avg_grad_norm.item() / gradient_accumulation_steps

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                t4 = time.time()

                data_loading_time += (t1 - t0) / gradient_accumulation_steps
                prepare_everything_time += (t2 - t1) / gradient_accumulation_steps
                network_forward_time += (t3 - t2) / gradient_accumulation_steps
                network_backward_time += (t4 - t3) / gradient_accumulation_steps

                t0 = time.time()
            
            ### <<<< Training <<<< ###
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if use_ema:
                    if not train_data.return_cond_videos:
                        ema_unet.step(unet.parameters())
                    else:
                        ema_controlnet.step(controlnet.parameters())
                progress_bar.update(1)
                global_step += 1

                # Wandb logging
                if accelerator.is_main_process and (not is_debug) and use_wandb:
                    wandb.log({"metrics/train_loss": train_loss}, step=global_step)
                    wandb.log({"metrics/train_grad_norm": train_grad_norm}, step=global_step)
                    
                    wandb.log({"profiling/train_data_loading_time": data_loading_time}, step=global_step)
                    wandb.log({"profiling/train_prepare_everything_time": prepare_everything_time}, step=global_step)
                    wandb.log({"profiling/train_network_forward_time": network_forward_time}, step=global_step)
                    wandb.log({"profiling/train_network_backward_time": network_backward_time}, step=global_step)
                    # accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                train_grad_norm = 0.0
                data_loading_time = 0.0
                prepare_everything_time = 0.0
                network_forward_time = 0.0
                network_backward_time = 0.0
                
                # Save checkpoint
                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if only_save_latest_checkpoint:
                            # Delete previous checkpoints
                            os.system(f"rm -rf {output_dir}/checkpoints/*")
                        
                        save_path = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path} (global_step: {global_step})")
                
                # Periodically validation
                if accelerator.is_main_process and global_step % validation_steps == 0:
                    if use_ema:
                        if not train_data.return_cond_videos:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        else:
                            # Store the ControlNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_controlnet.store(controlnet.parameters())
                            ema_controlnet.copy_to(controlnet.parameters())
                        
                    samples = []
                    wandb_samples = []
                    
                    generator = torch.Generator(device=latents.device)
                    generator.manual_seed(seed)
                    
                    prompts = validation_data.prompts
                    
                    first_frame_paths = [None] * len(prompts)
                    if unet_additional_kwargs["first_frame_condition_mode"] != "none":
                        first_frame_paths = validation_data.path_to_first_frames

                    validation_graph_embs = [None] * len(prompts)
                    validation_cond_video_paths = [None] * len(prompts)

                    if class_label_condition_mode:
                        for i_gr, val_gr in enumerate(train_dataset.validation_graphs_loaded):                        
                            val_gr = val_gr.to(latents.device)
                            val_gr_emb = get_embedding(val_gr, embedding_type=embedding_type).squeeze()
                            validation_graph_embs[i_gr] = val_gr_emb

                    if train_data.return_cond_videos:
                        validation_cond_video_paths = train_dataset.validation_cond_seq_paths

                    for idx, (prompt, first_frame_path, graph_emb, cond_video_path) in enumerate(zip(prompts, first_frame_paths, validation_graph_embs, validation_cond_video_paths)):
                        sample = validation_pipeline(
                            prompt,
                            generator    = generator,
                            video_length = train_data.sample_n_frames if not is_image else 2,
                            height       = height,
                            width        = width,
                            first_frame_paths = first_frame_path,
                            noise_sampling_method = unet_additional_kwargs['noise_sampling_method'],
                            noise_alpha = float(unet_additional_kwargs['noise_alpha']),
                            class_labels = graph_emb,
                            cond_video_frames_paths = cond_video_path,
                            **validation_data,
                        ).videos
                        # save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                        samples.append(sample)
                        
                        numpy_sample = (sample.squeeze(0).permute(1, 0, 2, 3) * 255).cpu().numpy().astype(np.uint8)
                        if (not is_debug) and use_wandb:
                            wandb_video = wandb.Video(numpy_sample, fps=8, caption=prompt)
                            wandb_samples.append(wandb_video)
                    
                    if (not is_debug) and use_wandb:
                        val_title = 'val_videos'
                        wandb.log({val_title: wandb_samples}, step=global_step)
                    
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)

                    logger.info(f"Saved samples to {save_path}")

                    if use_ema:
                        if not train_data.return_cond_videos:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        else:
                            # Switch back to the original ControlNet parameters.
                            ema_controlnet.restore(controlnet.parameters())
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if accelerator.is_main_process and (not is_debug) and use_wandb:
                wandb.log({"metrics/train_lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
            
            if global_step >= max_train_steps:
                break
            
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not train_data.return_cond_videos:
            unet = accelerator.unwrap_model(unet)
            pipeline = ConditionalAnimationPipeline(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=noise_scheduler,
            )
            pipeline.save_pretrained(f"{output_dir}/final_checkpoint")
        else:
            controlnet = accelerator.unwrap_model(controlnet)
            pipeline = ControlNetAnimationPipeline(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                controlnet=controlnet,
                tokenizer=tokenizer,
                scheduler=noise_scheduler,
            )
            pipeline.save_pretrained(f"{output_dir}/final_checkpoint")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--name", "-n", type=str, default="")
    parser.add_argument("--wandb",    action="store_true")
    parser.add_argument("optional_args", nargs='*', default=[])
    args = parser.parse_args()

    name   = args.name + "_" + Path(args.config).stem
    config = OmegaConf.load(args.config)

    if args.optional_args:
        modified_config = OmegaConf.from_dotlist(args.optional_args)
        config = OmegaConf.merge(config, modified_config)

    main(name=name, use_wandb=args.wandb, **config)
