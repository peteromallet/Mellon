import logging
logger = logging.getLogger('mellon')
import torch
from diffusers import SD3Transformer2DModel, AutoencoderKL, StableDiffusion3Pipeline
from utils.node_utils import NodeBase
from utils.memory_manager import memory_flush, memory_manager
from utils.hf_utils import is_local_files_only, get_repo_path
from utils.diffusers_utils import get_clip_prompt_embeds, get_t5_prompt_embeds
from config import config
from mellon.quantization import NodeQuantization
import math
import asyncio

HF_TOKEN = config.hf['token']

def calculate_mu(width: int, height: int, 
                patch_size: int = 2,
                base_image_seq_len: int = 256,
                max_image_seq_len: int = 4096,
                base_shift: float = 0.5,
                max_shift: float = 1.15) -> float:

    # latent size
    width = width // 8
    height = height // 8

    seq_len = (width // patch_size) * (height // patch_size)
    seq_len = max(min(seq_len, max_image_seq_len), base_image_seq_len)

    # this is the default mu calculation
    #m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    #b = base_shift - m * base_image_seq_len
    #mu = seq_len * m + b
    
    # This is my own mess. TODO: check if this is correct
    factor = (math.log2(seq_len) - math.log2(base_image_seq_len)) / (math.log2(max_image_seq_len) - math.log2(base_image_seq_len))
    factor = max(min(factor, 1.0), 0.0)
    mu = base_shift + factor * (max_shift - base_shift)

    return mu

class SD3UnifiedLoader(NodeBase):
    def execute(
            self,
            model_id,
            dtype,
            device,
            load_t5
        ):
        from modules.VAE.VAE import LoadVAE

        model_id = model_id or 'stabilityai/stable-diffusion-3.5-large'

        transformer = SD3TransformerLoader()(model_id=model_id, dtype=dtype, device=device)['model']
        transformer['transformer'] = self.mm_add(transformer['transformer'], priority=3)

        text_encoders = SD3TextEncodersLoader()(model_id=model_id, dtype=dtype, load_t5=load_t5, device=device)['model']
        text_encoders['text_encoder'] = self.mm_add(text_encoders['text_encoder'], priority=1)
        text_encoders['text_encoder_2'] = self.mm_add(text_encoders['text_encoder_2'], priority=1)
        text_encoders['t5_encoder'] = self.mm_add(text_encoders['t5_encoder'], priority=0) if load_t5 else None

        vae = LoadVAE()(model_id=model_id, device=device)['model']
        vae['vae'] = self.mm_add(vae['vae'], priority=2)

        return { 'model': {
            'transformer': transformer['transformer'],
            'text_encoder': text_encoders['text_encoder'],
            'tokenizer': text_encoders['tokenizer'],
            'text_encoder_2': text_encoders['text_encoder_2'],
            'tokenizer_2': text_encoders['tokenizer_2'],
            't5_encoder': text_encoders['t5_encoder'],
            't5_tokenizer': text_encoders['t5_tokenizer'],
            'vae': vae['vae'],
            'device': device,
            'model_id': model_id,
        }}

class SD3TransformerLoader(NodeBase, NodeQuantization):
    def execute(self, model_id, dtype, device, quantization, **kwargs):
        import os
        model_id = model_id or 'stabilityai/stable-diffusion-3.5-large'

        local_files_only = is_local_files_only(model_id)

        # overcome bug in diffusers loader with sharded weights
        if local_files_only:
            model_path = os.path.join(get_repo_path(model_id), "transformer")
        else:
            model_path = model_id

        transformer_model = SD3Transformer2DModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            subfolder="transformer" if not local_files_only else None,
            token=HF_TOKEN,
            local_files_only=local_files_only,
        )

        mm_id = self.mm_add(transformer_model, priority=3)

        if quantization != 'none':
            self.quantize(quantization, model=mm_id, device=device, **kwargs)

        return { 'model': { 'transformer': mm_id, 'device': device, 'model_id': model_id } }


class SD3TextEncodersLoader(NodeBase, NodeQuantization):
    def execute(self, model_id, dtype, load_t5, device, quantization, **kwargs):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

        model_id = model_id or 'stabilityai/stable-diffusion-3.5-large'

        model_cfg = {
            'torch_dtype': dtype,
            'token': HF_TOKEN,
            'local_files_only': is_local_files_only(model_id),
            #'use_safetensors': True,
        }

        text_encoder = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder", **model_cfg)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **model_cfg)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", **model_cfg)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", **model_cfg)
        text_encoder = self.mm_add(text_encoder, priority=1)
        text_encoder_2 = self.mm_add(text_encoder_2, priority=1)

        t5_encoder = None
        t5_tokenizer = None
        if load_t5:
            t5_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_3", **model_cfg)
            t5_tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3", **model_cfg)
            t5_encoder = self.mm_add(t5_encoder, priority=0)

        if quantization != 'none' and t5_encoder:
            self.quantize(quantization, model=t5_encoder, device=device, **kwargs)

        return { 'model': {
            'text_encoder': text_encoder,
            'tokenizer': tokenizer,
            'text_encoder_2': text_encoder_2,
            'tokenizer_2': tokenizer_2,
            't5_encoder': t5_encoder,
            't5_tokenizer': t5_tokenizer,
            'device': device,
            'model_id': model_id,
        }}

class SD3PromptEncoder(NodeBase):
    def execute(self, text_encoders, prompt, prompt_2, prompt_3, prompt_scale, prompt_scale_2, prompt_scale_3):
        return { 'embeds': self.encode_prompt(text_encoders, prompt, prompt_2, prompt_3, prompt_scale, prompt_scale_2, prompt_scale_3) }
    
    def encode_prompt(self, text_encoders, prompt="", prompt_2="", prompt_3="", prompt_scale=1.0, prompt_scale_2=1.0, prompt_scale_3=1.0, max_sequence_length=256):
        device = text_encoders['device']

        prompt = prompt or ""
        prompt_2 = prompt_2 or prompt
        prompt_3 = prompt_3 or prompt

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
        prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

        with torch.inference_mode():
            # 1. Get the embeds from the first text encoder
            encoder = self.mm_load(text_encoders['text_encoder'], device)
            prompt_embed, pooled_prompt_embed = get_clip_prompt_embeds(prompt, text_encoders['tokenizer'], encoder)
            if prompt_scale != 1.0:
                prompt_embed = prompt_embed * prompt_scale
                pooled_prompt_embed = pooled_prompt_embed * prompt_scale

            # 2. Get the embeds from the second text encoder
            encoder_2 = self.mm_load(text_encoders['text_encoder_2'], device)
            prompt_2_embed, pooled_prompt_2_embed = get_clip_prompt_embeds(prompt_2, text_encoders['tokenizer_2'], encoder_2)
            if prompt_scale_2 != 1.0:
                prompt_2_embed = prompt_2_embed * prompt_scale_2
                pooled_prompt_2_embed = pooled_prompt_2_embed * prompt_scale_2

            # 3. Get the concatenated clip embedings
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1).to('cpu')
            del prompt_embed, pooled_prompt_embed, prompt_2_embed, pooled_prompt_2_embed
            memory_flush()

            # 4. Get the T5 embedings
            if text_encoders['t5_encoder']:
                encoder_3 = self.mm_load(text_encoders['t5_encoder'], device)
                t5_prompt_embeds = get_t5_prompt_embeds(prompt_3, text_encoders['t5_tokenizer'], encoder_3)
                if prompt_scale_3 != 1.0:
                    t5_prompt_embeds = t5_prompt_embeds * prompt_scale_3
            else:
                t5_prompt_embeds = torch.zeros((clip_prompt_embeds.shape[0], max_sequence_length, 4096), device=device, dtype=clip_prompt_embeds.dtype)

        # 5. Merge the clip and T5 embedings
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2).to('cpu')

        del clip_prompt_embeds, t5_prompt_embeds # help the garbage collector?

        return { 'prompt_embeds': prompt_embeds, 'pooled_prompt_embeds': pooled_prompt_embeds }

class SD3SimplePromptEncoder(SD3PromptEncoder):
    def execute(self, text_encoders, prompt, negative_prompt):
        return { 'embeds': {
            'positive_embeds': self.encode_prompt(text_encoders, prompt),
            'negative_embeds': self.encode_prompt(text_encoders, negative_prompt),
        }}

class SD3Sampler(NodeBase):
    def __init__(self, node_id):
        super().__init__(node_id)

        self.dummy_vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            latent_channels=16,
        )

        self.schedulers_config = {
            'FlowMatchEulerDiscreteScheduler': {
                'num_train_timesteps': 1000,
                'shift': 3.0,
                'use_dynamic_shifting': False,
                'base_shift': 0.5,
                'max_shift': 1.15,
                'base_image_seq_len': 256,
                'max_image_seq_len': 4096,
                'invert_sigmas': False,
            },
            'FlowMatchHeunDiscreteScheduler': {
                'num_train_timesteps': 1000,
                'shift': 3.0,
            }
        }
        
    def execute(self,
                transformer,
                prompt,
                negative_prompt,
                latents_in,
                seed,
                scheduler,
                width,
                height,
                steps,
                cfg,
                shift,
                use_dynamic_shifting):
        
        # 1. Prepare the pipeline
        device = transformer['device']
        transformer_model = self.mm_load(transformer['transformer'], device)
        model_id = transformer['model_id']
        generator = torch.Generator(device=device).manual_seed(seed)

        vae = self.mm_flash_load(self.dummy_vae, device=device)

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=transformer_model,
            text_encoder=None,
            text_encoder_2=None,
            text_encoder_3=None,
            tokenizer=None,
            tokenizer_2=None,
            tokenizer_3=None,
            scheduler=None,
            vae=vae,
            local_files_only=True,
        )

        # 2. Create the scheduler
        if scheduler == 'FlowMatchHeunDiscreteScheduler':
            from diffusers import FlowMatchHeunDiscreteScheduler as SchedulerCls
            use_dynamic_shifting = False # not supported by Heun
        else:
            from diffusers import FlowMatchEulerDiscreteScheduler as SchedulerCls

        scheduler_config = self.schedulers_config[scheduler]
        mu = None
        if use_dynamic_shifting:
            mu = calculate_mu(width, height,
                            patch_size=transformer_model.config.patch_size,
                            base_image_seq_len=scheduler_config['base_image_seq_len'],
                            max_image_seq_len=scheduler_config['max_image_seq_len'],
                            base_shift=scheduler_config['base_shift'],
                            max_shift=scheduler_config['max_shift'])

        pipe.scheduler = SchedulerCls.from_config(scheduler_config, shift=shift, use_dynamic_shifting=use_dynamic_shifting)

        """
        batch_size = positive['prompt_embeds'].shape[0]
        num_images_per_prompt = 1 # TODO: make this dynamic
        num_channels_latents = pipe.transformer.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            positive['prompt_embeds'].dtype,
            transformer_model.device,
            generator,
            latents_in,
        )
        """

        # 3. Prepare the prompts
        if 'positive_embeds' in prompt:
            positive = prompt['positive_embeds']
            negative = prompt['negative_embeds']
        else:
            positive = prompt
            negative = negative_prompt

        if not negative:
            negative = { 'prompt_embeds': torch.zeros_like(positive['prompt_embeds']), 'pooled_prompt_embeds': torch.zeros_like(positive['pooled_prompt_embeds']) }
        
        def pad_prompt_embeds(source, target):
            pad_length = target['prompt_embeds'].shape[1] - source['prompt_embeds'].shape[1]
            source['prompt_embeds'] = torch.nn.functional.pad(source['prompt_embeds'], (0, 0, 0, pad_length))
            return source

        if positive['prompt_embeds'].shape[1] > negative['prompt_embeds'].shape[1]:
            negative = pad_prompt_embeds(negative, positive)
        elif positive['prompt_embeds'].shape[1] < negative['prompt_embeds'].shape[1]:
            positive = pad_prompt_embeds(positive, negative)

        # 4. Run the pipeline
        with torch.inference_mode():
            while True:
                try:
                    latents = pipe(
                        generator=generator,
                        prompt_embeds=positive['prompt_embeds'].to(device, dtype=transformer_model.dtype),
                        pooled_prompt_embeds=positive['pooled_prompt_embeds'].to(device, dtype=transformer_model.dtype),
                        negative_prompt_embeds=negative['prompt_embeds'].to(device, dtype=transformer_model.dtype),
                        negative_pooled_prompt_embeds=negative['pooled_prompt_embeds'].to(device, dtype=transformer_model.dtype),
                        #latents=latents,
                        width=width,
                        height=height,
                        guidance_scale=cfg,
                        num_inference_steps=steps,
                        output_type="latent",
                        callback_on_step_end=self.pipe_callback,
                        mu=mu,
                    ).images
                    break
                except torch.OutOfMemoryError as e:
                    if memory_manager.unload_next(device, exclude=transformer['transformer']):
                        continue
                    else:
                        raise e
                except Exception as e:
                    raise e

        latents = latents.to('cpu')

        del pipe
        memory_flush()

        return { 'latents': latents }
