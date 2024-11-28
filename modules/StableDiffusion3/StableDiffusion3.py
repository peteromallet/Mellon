import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline, AutoencoderKL
from utils.node_utils import NodeBase
from utils.memory_manager import memory_flush

class LoadSD3Transformer(NodeBase):
    def execute(self, model_id, dtype, device):
        model = SD3Transformer2DModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            subfolder="transformer",
        )

        model = self.mm_add(model, priority='high')
 
        return { 'transformer': { 'model': model, 'device': device } }


class SD3TextEncodersLoader(NodeBase):
    def execute(self, model_id, dtype, load_t5, device):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

        text_encoder = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=dtype)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", torch_dtype=dtype)
        text_encoder = self.mm_add(text_encoder, priority='low')
        text_encoder_2 = self.mm_add(text_encoder_2, priority='low')

        t5_encoder = None
        t5_tokenizer = None
        if load_t5:
            t5_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_3", torch_dtype=dtype)
            t5_tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3", torch_dtype=dtype)
            t5_encoder = self.mm_add(t5_encoder, priority='low')

        return { 'text_encoders': {
            'text_encoder': text_encoder,
            'tokenizer': tokenizer,
            'text_encoder_2': text_encoder_2,
            'tokenizer_2': tokenizer_2,
            't5_encoder': t5_encoder,
            't5_tokenizer': t5_tokenizer,
            'device': device
        } }

class SD3TextEncoder(NodeBase):
    def execute(self, text_encoders, t5_encoders, prompt, prompt_2, prompt_3, prompt_scale, prompt_scale_2, prompt_scale_3):
        if 't5_encoder' in text_encoders:
            t5_encoders = { 't5_encoder': text_encoders['t5_encoder'], 't5_tokenizer': text_encoders['t5_tokenizer'], 'device': text_encoders['device'] }

        clip_encoders = [text_encoders['text_encoder'], text_encoders['text_encoder_2']]
        clip_tokenizers = [text_encoders['tokenizer'], text_encoders['tokenizer_2']]

        t5_encoder = t5_encoders['t5_encoder'] if t5_encoders else None
        t5_tokenizer = t5_encoders['t5_tokenizer'] if t5_encoders else None

        device_clip = text_encoders['device']
        device_t5 = t5_encoders['device'] if t5_encoders else None

        prompt = prompt or ""
        prompt_2 = prompt_2 or prompt
        prompt_3 = prompt_3 or prompt

        # Encode the CLIP prompts
        clip_embeds = []
        clip_pooled_embeds = []
        clip_prompts = [prompt, prompt_2]
        prompt_scales = [prompt_scale, prompt_scale_2]
        with torch.inference_mode():
        #with torch.no_grad():
            for i, prompt in enumerate(clip_prompts):
                encoder = self.mm_load(clip_encoders[i], device_clip)
                prompt = [prompt] if isinstance(prompt, str) else prompt
                scale = prompt_scales[i]
                batch_size = len(prompt)

                text_input_ids = clip_tokenizers[i](
                    prompt,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids

                embeds = encoder(text_input_ids.to(device_clip), output_hidden_states=True)
                pooled_embeds = embeds[0]
                embeds = embeds.hidden_states[-2]

                _, seq_len, _ = embeds.shape
                embeds = embeds.view(batch_size, seq_len, -1).to('cpu')
                pooled_embeds = pooled_embeds.view(batch_size, -1).to('cpu')

                if scale != 1.0:
                    embeds *= scale
                    pooled_embeds *= scale
                            
                clip_embeds.append(embeds)
                clip_pooled_embeds.append(pooled_embeds)

        del encoder, text_input_ids, embeds, pooled_embeds, clip_tokenizers
    
        clip_embeds = torch.cat(clip_embeds, dim=-1)
        pooled_embeds = torch.cat(clip_pooled_embeds, dim=-1)

        # Encode the T5 prompt
        if t5_encoder is None:
            t5_embeds = torch.zeros((batch_size, 256, 4096), device='cpu', dtype=clip_embeds.dtype)
        else:
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3
            batch_size = len(prompt_3)

            encoder = self.mm_load(t5_encoder, device_t5)
            text_input_ids = t5_tokenizer(
                prompt_3,
                padding="max_length",
                max_length=256,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids

            #with torch.no_grad():
            with torch.inference_mode():
                t5_embeds = encoder(text_input_ids.to(device_t5))[0]

            _, seq_len, _ = t5_embeds.shape
            t5_embeds = t5_embeds.view(batch_size, seq_len, -1).to('cpu', dtype=clip_embeds.dtype)

            if prompt_scale_3 != 1.0:
                t5_embeds *= prompt_scale_3

            del encoder, text_input_ids, t5_tokenizer

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

        transformer_model = self.mm_load(transformer['model'], transformer['device'])
        model_id = transformer_model.config['_name_or_path']
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
            transformer=transformer_model,
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

        #if not negative:
        #    negative = { 'embeds': torch.zeros_like(positive['embeds']), 'pooled_embeds': torch.zeros_like(positive['pooled_embeds']) }

        with torch.inference_mode():
        #with torch.no_grad():
            latents = pipe(
                generator=torch.Generator().manual_seed(seed),
                prompt_embeds=positive['embeds'].to(device, dtype=transformer_model.dtype),
                pooled_prompt_embeds=positive['pooled_embeds'].to(device, dtype=transformer_model.dtype),
                negative_prompt_embeds=negative['embeds'].to(device, dtype=transformer_model.dtype),
                negative_pooled_prompt_embeds=negative['pooled_embeds'].to(device, dtype=transformer_model.dtype),
                width=width,
                height=height,
                guidance_scale=cfg,
                num_inference_steps=steps,
                output_type="latent",
            ).images
        
        del pipe, positive, negative, transformer_model
        latents = latents.to('cpu')
        memory_flush()

        return { 'latents': latents }
