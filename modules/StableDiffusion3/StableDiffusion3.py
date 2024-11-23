import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline, AutoencoderKL
from utils.node_utils import NodeBase

class LoadSD3Transformer(NodeBase):
    def execute(self, model_id, dtype, device):
        model = SD3Transformer2DModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            subfolder="transformer",
        )
 
        return { 'transformer': { 'model': model, 'device': device } }


class SD3TextEncoder(NodeBase):
    def execute(self, text_encoders, t5_encoders, prompt, prompt_2, prompt_3):
        clip_encoders = [text_encoders['text_encoder'], text_encoders['text_encoder_2']]
        clip_tokenizers = [text_encoders['tokenizer'], text_encoders['tokenizer_2']]

        t5_encoder = t5_encoders['t5_encoder'] if t5_encoders else None
        t5_tokenizer = t5_encoders['t5_tokenizer'] if t5_encoders else None

        device_clip = text_encoders['device']
        device_t5 = t5_encoders['device'] if t5_encoders else None

        prompt = prompt or ""
        prompt_2 = prompt_2 or prompt
        prompt_3 = prompt_3 or prompt

        clip_prompts = [prompt, prompt_2]

        # start with the T5 prompt
        prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3
        batch_size = len(prompt_3)

        if t5_encoder is None:
            t5_embeds = None
        else:
            t5_encoder = t5_encoder.to(device_t5)
            with torch.inference_mode():
            #with torch.no_grad():
                text_input_ids = t5_tokenizer(
                    prompt_3,
                    padding="max_length",
                    max_length=256,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).input_ids.to(device_t5)

                t5_embeds = t5_encoder(text_input_ids)[0]

            _, seq_len, _ = t5_embeds.shape
            t5_embeds = t5_embeds.view(batch_size, seq_len, -1)
            t5_embeds = t5_embeds.to('cpu', dtype=t5_encoder.dtype)

            t5_encoder.to('cpu')
            text_input_ids.to('cpu')
            t5_encoder = None
            text_input_ids = None
            t5_tokenizer = None
            torch.cuda.empty_cache()

        # now the CLIP prompts
        clip_embeds = []
        clip_pooled_embeds = []
        with torch.inference_mode():
        #with torch.no_grad():
            for i, prompt in enumerate(clip_prompts):
                clip_encoder = clip_encoders[i].to(device_clip)
                prompt = [prompt] if isinstance(prompt, str) else prompt
                batch_size = len(prompt)

                text_input_ids = clip_tokenizers[i](
                    prompt,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids

                embeds = clip_encoder(text_input_ids.to(device_clip), output_hidden_states=True)
                pooled_embeds = embeds[0].to('cpu', dtype=clip_encoder.dtype)
                embeds = embeds.hidden_states[-2].to('cpu', dtype=clip_encoder.dtype)

                _, seq_len, _ = embeds.shape
                embeds = embeds.view(batch_size, seq_len, -1)
                pooled_embeds = pooled_embeds.view(batch_size, -1)
                            
                clip_embeds.append(embeds)
                clip_pooled_embeds.append(pooled_embeds)

                clip_encoder.to('cpu')
                clip_encoder = None
                text_input_ids = None
                embeds = None
                pooled_embeds = None
    
        del clip_encoders, clip_tokenizers
        torch.cuda.empty_cache()
        clip_embeds = torch.cat(clip_embeds, dim=-1)
        pooled_embeds = torch.cat(clip_pooled_embeds, dim=-1)

        if t5_embeds is None:
            t5_embeds = torch.zeros((batch_size, 256, 4096), dtype=clip_embeds.dtype)

        clip_embeds = torch.nn.functional.pad(
            clip_embeds, (0, t5_embeds.shape[-1] - clip_embeds.shape[-1])
        )
        embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)

        return { 'embeds': { 'embeds': embeds, 'pooled_embeds': pooled_embeds } }

class SD3Sampler(NodeBase):
    def execute(self,
                transformer,
                positive,
                negative,
                latents_in,
                seed,
                width,
                height,
                steps,
                cfg):
        
        model_id = transformer['model'].config['_name_or_path']
        device = transformer['device']
        #torch.backends.cudnn.allow_tf32 = False
        #torch.backends.cuda.matmul.allow_tf32 = False

        dummy_vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            latent_channels=16,
        )

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=transformer['model'],
            text_encoder=None,
            text_encoder_2=None,
            text_encoder_3=None,
            tokenizer=None,
            tokenizer_2=None,
            tokenizer_3=None,
            vae=dummy_vae
        ).to(device)

        #pipe.enable_xformers_memory_efficient_attention()
        #pipe.enable_model_cpu_offload()

        with torch.inference_mode():
            latents = pipe(
                generator=torch.Generator().manual_seed(seed),
                prompt_embeds=positive['embeds'].to(device, dtype=transformer['model'].dtype),
                pooled_prompt_embeds=positive['pooled_embeds'].to(device, dtype=transformer['model'].dtype),
                negative_prompt_embeds=negative['embeds'].to(device, dtype=transformer['model'].dtype),
                negative_pooled_prompt_embeds=negative['pooled_embeds'].to(device, dtype=transformer['model'].dtype),
                width=width,
                height=height,
                guidance_scale=cfg,
                num_inference_steps=steps,
                output_type="latent",
            ).images
        
        pipe.to('cpu')
        latents = latents.to('cpu')
        del pipe, positive
        torch.cuda.empty_cache()

        return { 'latents': latents }
