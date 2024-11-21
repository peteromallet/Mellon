import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from utils.torch_utils import device_list, toPIL
from utils.node_utils import NodeBase

class LoadUNet(NodeBase):
    def execute(self, model_id, variant, dtype, use_safetensors, device):
        model = UNet2DConditionModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant=variant,
            use_safetensors=use_safetensors,
            subfolder="unet",
        )
 
        return { 'unet': { 'model': model, 'device': device }}

class LoadTextEncoder(NodeBase):
    def execute(self, model_id, device):
        # This is a CLIP encoder helper specific to the Stable Diffusion pipeline, 
        # we assume the encoder to be clip-vit-large-patch14 otherwise we try to
        # load the models from the subfolder of the main model_id repository.

        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder" if not model_id.endswith("clip-vit-large-patch14") else '')
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer" if not model_id.endswith("clip-vit-large-patch14") else '')

        return { 'clip': { 'text_encoder': text_encoder, 'tokenizer': tokenizer, 'device': device } }

class LoadVAE(NodeBase): 
    def execute(self, model_id, device):
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

        return { 'vae': { 'model': vae, 'device': device } }

class TextEncoder(NodeBase):
    def execute(self, prompt, clip):
        device = clip['device']
        text_encoder = clip['text_encoder'].to(device)
        inputs = clip['tokenizer'](prompt, return_tensors="pt").input_ids.to(device)

        #with torch.no_grad(): which one!?
        with torch.inference_mode():
            text_embeddings = text_encoder(inputs).last_hidden_state

        # clean up
        clip['text_encoder'] = clip['text_encoder'].to('cpu')
        inputs = inputs.to('cpu')
        inputs = None

        return { 'embeds': text_embeddings.cpu() }

class SDSampler(NodeBase):
    def execute(self, unet, positive, negative, seed, latents_in, width, height, steps, cfg):
        # TODO: add support for latents_in

        model_id = unet['model'].config['_name_or_path']
        device = unet['device']

        dummy_vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            latent_channels=4,
        ) # TODO: do we need any more values for a dummy vae?

        # TODO: does this really load only the config?
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            unet=unet['model'],
            text_encoder=None,
            tokenizer=None,
            vae=dummy_vae
        ).to(device)

        #latents_in = latents_in.to(unet['device'])

        # Pad embeddings to same length
        # TODO: am I doing this right? should we pad to maximum length?
        if negative is not None and positive.shape[1] != negative.shape[1]:
            max_length = max(positive.shape[1], negative.shape[1])
            if positive.shape[1] < max_length:
                positive = torch.nn.functional.pad(positive, (0, 0, 0, max_length - positive.shape[1]))
            else:
                negative = torch.nn.functional.pad(negative, (0, 0, 0, max_length - negative.shape[1]))

        #with torch.no_grad():
        with torch.inference_mode():
            latents = pipe(
                generator=torch.Generator(device=device).manual_seed(seed),
                prompt_embeds=positive.to(device),
                negative_prompt_embeds=negative.to(device) if negative is not None else None,
                #latents=latents_in,
                width=width,
                height=height,
                guidance_scale=cfg,
                num_inference_steps=steps,
                output_type="latent",
            ).images

        #pipe = pipe.to('cpu')
        latents = latents.to('cpu')
        del pipe, positive, negative

        return { 'latents': latents }
