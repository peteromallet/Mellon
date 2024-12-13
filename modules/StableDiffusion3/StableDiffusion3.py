import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline, AutoencoderKL
from utils.node_utils import NodeBase
from utils.memory_manager import memory_flush
from importlib import import_module
from config import config

HF_TOKEN = config.hf['token']

class LoadSD3Transformer(NodeBase):
    def execute(self, model_id, dtype, device):
        model = SD3Transformer2DModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            subfolder="transformer",
            token=HF_TOKEN,
        )

        model = self.mm_add(model, priority=3)
 
        return { 'transformer': { 'model': model, 'device': device } }


class SD3TextEncodersLoader(NodeBase):
    def execute(self, model_id, dtype, load_t5, device):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

        text_encoder = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype, token=HF_TOKEN)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=dtype, token=HF_TOKEN)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype, token=HF_TOKEN)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", torch_dtype=dtype, token=HF_TOKEN)
        text_encoder = self.mm_add(text_encoder, priority=1)
        text_encoder_2 = self.mm_add(text_encoder_2, priority=1)

        t5_encoder = None
        t5_tokenizer = None
        if load_t5:
            t5_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_3", torch_dtype=dtype, token=HF_TOKEN)
            t5_tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3", torch_dtype=dtype, token=HF_TOKEN)
            t5_encoder = self.mm_add(t5_encoder, priority=1)

        return { 'text_encoders': {
            'text_encoder': text_encoder,
            'tokenizer': tokenizer,
            'text_encoder_2': text_encoder_2,
            'tokenizer_2': tokenizer_2,
            't5_encoder': t5_encoder,
            't5_tokenizer': t5_tokenizer,
            'device': device
        }}

class SD3PromptEncoder(NodeBase):
    def execute(self, text_encoders, prompt, prompt_2, prompt_3, prompt_scale, prompt_scale_2, prompt_scale_3):        
        model_id = getattr(self.mm_get(text_encoders['text_encoder']).config, '_name_or_path')
        device = text_encoders['device']
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            local_files_only=True,
            transformer=None,
            vae=None,
            scheduler=None,
            text_encoder=self.mm_get(text_encoders['text_encoder']),
            text_encoder_2=self.mm_get(text_encoders['text_encoder_2']),
            text_encoder_3=self.mm_get(text_encoders['t5_encoder']) if text_encoders['t5_encoder'] else None,
            tokenizer=text_encoders['tokenizer'],
            tokenizer_2=text_encoders['tokenizer_2'],
            tokenizer_3=text_encoders['t5_tokenizer'],
        )

        prompt = prompt or ""
        prompt_2 = prompt_2 or prompt
        prompt_3 = prompt_3 or prompt

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
        prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

        with torch.inference_mode():
            self.mm_load(text_encoders['text_encoder'], device)
            prompt_embed, pooled_prompt_embed = pipe._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                clip_skip=None,
                clip_model_index=0,
            )
            if prompt_scale != 1.0:
                prompt_embed = prompt_embed * prompt_scale
                pooled_prompt_embed = pooled_prompt_embed * prompt_scale

            self.mm_load(text_encoders['text_encoder_2'], device)
            prompt_2_embed, pooled_prompt_2_embed = pipe._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=1,
                clip_skip=None,
                clip_model_index=1,
            )
            if prompt_scale_2 != 1.0:
                prompt_2_embed = prompt_2_embed * prompt_scale_2
                pooled_prompt_2_embed = pooled_prompt_2_embed * prompt_scale_2

            t5_prompt_embed = None
            if text_encoders['t5_encoder']:
                self.mm_load(text_encoders['t5_encoder'], device)
                t5_prompt_embed = pipe._get_t5_prompt_embeds(
                    prompt=prompt_3,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=256,
                )
                if prompt_scale_3 != 1.0:
                    t5_prompt_embed = t5_prompt_embed * prompt_scale_3
            else:
                t5_prompt_embed = torch.zeros((prompt_embed.shape[0], 256, 4096), device=device, dtype=prompt_embed.dtype)

        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2).to('cpu')

        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1).to('cpu')

        del pipe, prompt_embed, pooled_prompt_embed, prompt_2_embed, pooled_prompt_2_embed, t5_prompt_embed
        memory_flush()

        return { 'embeds': { 'prompt_embeds': prompt_embeds, 'pooled_prompt_embeds': pooled_prompt_embeds } }

class SD3Sampler(NodeBase):
    def execute(self,
                transformer,
                positive,
                negative,
                latents_in,
                seed,
                scheduler,
                width,
                height,
                steps,
                cfg):

        device = transformer['device']        
        transformer_model = self.mm_load(transformer['model'], device)
        model_id = transformer_model.config['_name_or_path']

        dummy_vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            latent_channels=16,
        )

        scheduler_config = {
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
        scheduler_cls = getattr(import_module("diffusers"), scheduler)
        scheduler_cls = scheduler_cls.from_config(scheduler_config[scheduler])

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=transformer_model,
            text_encoder=None,
            text_encoder_2=None,
            text_encoder_3=None,
            tokenizer=None,
            tokenizer_2=None,
            tokenizer_3=None,
            scheduler=scheduler_cls,
            vae=dummy_vae,
            local_files_only=True,
        ).to(device)

        if not negative:
            negative = { 'prompt_embeds': torch.zeros_like(positive['prompt_embeds']), 'pooled_prompt_embeds': torch.zeros_like(positive['pooled_prompt_embeds']) }

        with torch.inference_mode():
        #with torch.no_grad():
            latents = pipe(
                generator=torch.Generator('cpu').manual_seed(seed),
                prompt_embeds=positive['prompt_embeds'].to(device, dtype=transformer_model.dtype),
                pooled_prompt_embeds=positive['pooled_prompt_embeds'].to(device, dtype=transformer_model.dtype),
                negative_prompt_embeds=negative['prompt_embeds'].to(device, dtype=transformer_model.dtype),
                negative_pooled_prompt_embeds=negative['pooled_prompt_embeds'].to(device, dtype=transformer_model.dtype),
                width=width,
                height=height,
                guidance_scale=cfg,
                num_inference_steps=steps,
                output_type="latent",
            ).images

        latents = latents.to('cpu')

        del pipe, positive, negative, transformer_model, scheduler_cls
        memory_flush()

        return { 'latents': latents }
