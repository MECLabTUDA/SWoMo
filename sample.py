import argparse
import numpy as np
import imageio
import os
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from einops import rearrange
from torch_geometric.loader import DataLoader
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from swomo.models.videoldm_unet import VideoLDMUNet3DConditionModel
from swomo.models.videoldm_controlnet import ControlNetModel
from swomo.pipelines.pipeline_conditional_animation import ConditionalAnimationPipeline
from swomo.pipelines.pipeline_controlnet_animation import ControlNetAnimationPipeline
from swomo.data.dataset import SurgicalDataset
from swomo.graph_encoder.graph_segclip_masked import GraphEncoder

def main(args, config):
    savedir = f"{config.output_dir}"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    class_embeddings_concat = (config.unet_graph_kwargs.get('class_embeddings_concat') if config.unet_graph_kwargs is not None else False)
    class_embed_type = (config.unet_graph_kwargs.get('class_embed_type') if config.unet_graph_kwargs is not None else None)
    time_embedding_dim = (config.unet_graph_kwargs.get('time_embedding_dim') if config.unet_graph_kwargs is not None else None)
    ignore_mismatched_sizes = (config.unet_graph_kwargs.get('ignore_mismatched_sizes') if config.unet_graph_kwargs is not None else False)

    ### >>> create validation pipeline >>> ###
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(config.noise_scheduler_kwargs))
    tokenizer       = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer", use_safetensors=True)
    text_encoder    = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
    vae             = AutoencoderKL.from_pretrained(config.finetuned_autoencoder_path if config.finetuned_autoencoder_path else config.pretrained_model_path, subfolder="vae", use_safetensors=True)
    unet            = VideoLDMUNet3DConditionModel.from_pretrained(
        config.pretrained_model_path,
        subfolder="unet",
        variant=config.unet_additional_kwargs['variant'],
        use_temporal=True,
        temp_pos_embedding=config.unet_additional_kwargs['temp_pos_embedding'],
        augment_temporal_attention=config.unet_additional_kwargs['augment_temporal_attention'],
        n_frames=config.sampling_kwargs['n_frames'],
        n_temp_heads=config.unet_additional_kwargs['n_temp_heads'],
        first_frame_condition_mode=config.unet_additional_kwargs['first_frame_condition_mode'],
        use_frame_stride_condition=config.unet_additional_kwargs['use_frame_stride_condition'],

        class_embeddings_concat=class_embeddings_concat,
        class_embed_type=class_embed_type,
        time_embedding_dim=time_embedding_dim,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        use_safetensors=True
    )

    if config.controlnet_path is not None:
        controlnet = ControlNetModel.from_unet(unet)

    # 1. unet ckpt
    if config.unet_path is not None:
        if os.path.isdir(config.unet_path):
            unet_dict = VideoLDMUNet3DConditionModel.from_pretrained(config.unet_path)
            m, u = unet.load_state_dict(unet_dict.state_dict(), strict=False)
            assert len(u) == 0
            del unet_dict
        else:
            checkpoint_dict = torch.load(config.unet_path, map_location="cpu")
            state_dict = checkpoint_dict["state_dict"] if "state_dict" in checkpoint_dict else checkpoint_dict
            if config.unet_ckpt_prefix is not None:
                state_dict = {k.replace(config.unet_ckpt_prefix, ''): v for k, v in state_dict.items()}
            m, u = unet.load_state_dict(state_dict, strict=False)
            assert len(u) == 0

    if config.controlnet_path is not None:
        if os.path.isdir(config.controlnet_path):
            controlnet_dict = ControlNetModel.from_pretrained(config.controlnet_path)
            m, u = controlnet.load_state_dict(controlnet_dict.state_dict(), strict=False)
            assert len(u) == 0
            del controlnet_dict

        else:
            raise NotImplementedError("ControlNet loading from non-directory path is not implemented yet. Please provide a directory path for ControlNet checkpoint.")


    if is_xformers_available() and int(torch.__version__.split(".")[0]) < 2:
        unet.enable_xformers_memory_efficient_attention()
        if config.controlnet_path is not None:
            controlnet.enable_xformers_memory_efficient_attention()

    if config.controlnet_path is None:
        pipeline = ConditionalAnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=noise_scheduler)

    else:
        pipeline = ControlNetAnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet, scheduler=noise_scheduler,
        )
    pipeline.to("cuda")

    if config.test_data.return_graph_emb:
        if 'model_masked' in config.test_data:
            m_graph_emb_masked = GraphEncoder(config.test_data.graph_input_dim, config.test_data.graph_hidden_dim, 
                                                config.test_data.graph_embedding_dim, config.test_data.trainable,
                                                graph_conv_type = config.test_data.graph_conv_type, 
                                                graph_norm_type = config.test_data.graph_norm_type, 
                                                graph_encoder_ckpt = config.test_data.model_masked)
            m_graph_emb_masked.to("cuda")
            m_graph_emb_masked.eval()
        if 'model_segclip' in config.test_data:
            m_graph_emb_segclip = GraphEncoder(config.test_data.graph_input_dim, config.test_data.graph_hidden_dim, 
                                                config.test_data.graph_embedding_dim, config.test_data.trainable, 
                                                graph_conv_type = config.test_data.graph_conv_type, 
                                                graph_norm_type = config.test_data.graph_norm_type, 
                                                graph_encoder_ckpt = config.test_data.model_segclip)
            m_graph_emb_segclip.to("cuda")
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
    
    # (frameinit) initialize frequency filter for noise reinitialization -------------
    if config.frameinit_kwargs.enable:
        pipeline.init_filter(
            width         = config.sampling_kwargs.width,
            height        = config.sampling_kwargs.height,
            video_length  = config.sampling_kwargs.n_frames,
            filter_params = config.frameinit_kwargs.filter_params,
        )
    # -------------------------------------------------------------------------------
    ### <<< create validation pipeline <<< ###

    test_dataset = SurgicalDataset(**config.test_data)
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    resume_sampling_idx = getattr(config, "resume_sampling", 0) or 0
    print(f"Length of test dataloader: {len(test_dataloader)}")
    print(f"Resuming sampling from index {resume_sampling_idx}")

    for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Sampling SWoMo:"):
        
        if idx >= resume_sampling_idx:
            prompt = batch["text"]
            
            if config.test_data.return_graph_emb:
                graph = batch.get('graph').to("cuda")
                graph_emb = get_embedding(graph, config.test_data.embedding_type)
                graph_emb = graph_emb.detach().squeeze()
            else:
                graph_emb = None

            first_frame_paths = None
            if config.unet_additional_kwargs['first_frame_condition_mode'] != "none":
                first_frame_paths = batch["first_frame_path"]
            
            cond_video_paths =  [None] * len(prompt)
            if config.controlnet_path is not None and config.test_data.return_cond_videos:
                cond_video_paths = batch["cond_video_path"]
                cond_video_paths = [[cond_video_paths[i][0] for i in range(len(cond_video_paths))]]

            sample = pipeline(
                prompt,
                first_frame_paths       = first_frame_paths,
                cond_video_frames_paths = cond_video_paths,
                num_inference_steps     = config.sampling_kwargs.steps,
                guidance_scale_txt      = config.sampling_kwargs.guidance_scale_txt,
                guidance_scale_img      = config.sampling_kwargs.guidance_scale_img,
                width                   = config.sampling_kwargs.width,
                height                  = config.sampling_kwargs.height,
                video_length            = config.sampling_kwargs.n_frames,
                noise_sampling_method   = config.unet_additional_kwargs['noise_sampling_method'],
                noise_alpha             = float(config.unet_additional_kwargs['noise_alpha']),
                class_labels            = graph_emb,
                eta                     = config.sampling_kwargs.ddim_eta,
                frame_stride            = config.sampling_kwargs.frame_stride,
                guidance_rescale        = config.sampling_kwargs.guidance_rescale,
                num_videos_per_prompt   = config.sampling_kwargs.num_videos_per_prompt,
                use_frameinit           = config.frameinit_kwargs.enable,
                frameinit_noise_level   = config.frameinit_kwargs.noise_level,
                camera_motion           = config.frameinit_kwargs.camera_motion,
            ).videos

            current_batch = len(prompt)
            sample = rearrange(sample, "b c t h w -> b t h w c")
            
            for bc in range(current_batch):
                save_folder = os.path.join(config.output_dir, str(idx + bc).zfill(5))
                os.makedirs(save_folder, exist_ok=True)
                seq = sample[bc]

                for i, img in enumerate(seq):
                    img = (img * 255).numpy().astype(np.uint8)
                    imageio.imsave(os.path.join(save_folder, str(i).zfill(2)+".png"), img)
        else: 
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, default="configs/inference/inference.yaml")
    parser.add_argument("optional_args", nargs='*', default=[])
    args = parser.parse_args()

    config = OmegaConf.load(args.inference_config)
    if args.optional_args:
        modified_config = OmegaConf.from_dotlist(args.optional_args)
        config = OmegaConf.merge(config, modified_config)

    main(args, config)
