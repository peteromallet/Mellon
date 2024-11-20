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
 
        self.output['unet'] = { 'model': model, 'device': device }

class LoadTextEncoder(NodeBase):
    def execute(self, model_id, device):
        # This is a CLIP encoder helper specific to the Stable Diffusion pipeline, 
        # we assume the encoder to be clip-vit-large-patch14 otherwise we try to
        # load the models from the subfolder of the main model_id repository.

        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder" if not model_id.endswith("clip-vit-large-patch14") else '')
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer" if not model_id.endswith("clip-vit-large-patch14") else '')

        self.output['clip'] = { 'text_encoder': text_encoder, 'tokenizer': tokenizer, 'device': device }

class LoadVAE(NodeBase): 
    def execute(self, model_id, device):
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

        self.output['vae'] = { 'model': vae, 'device': device }

class TextEncoder(NodeBase):
    def execute(self, prompt, clip):
        device = clip['device']
        text_encoder = clip['text_encoder'].to(device)
        inputs = clip['tokenizer'](prompt, return_tensors="pt").input_ids.to(device)

        #with torch.no_grad(): which one!?
        with torch.inference_mode():
            text_embeddings = text_encoder(inputs).last_hidden_state

        self.output['embeds'] = text_embeddings.cpu()

        # clean up
        clip['text_encoder'] = clip['text_encoder'].to('cpu')
        inputs = inputs.to('cpu')
        inputs = None
        text_embeddings = None

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

        self.output['latents'] = latents

class LoadModel():   
    def __init__(self):
        #self.model_id = None
        #self.variant = "fp16"
        #self.dtype = "Auto"
        #self.use_safetensors = True
        #self.safety_checker = None
        #self.add_watermarker = False
        self.params = {
            'model_id': None,
            'variant': None,
            'dtype': None,
            'use_safetensors': None,
        }

        #module_name = __name__.split('.')[-2]
        #class_name = self.__class__.__name__
        #params = MODULE_MAP[module_name][class_name]['params']
        #self.params = { param: None for param in params if param['display'] != 'output' and param['display'] != 'ui' }
        self.pipeline = None

        #for param in params:
            #defaultValue = None
            #if 'default' in param:
            #    defaultValue = param['default']
            #elif 'validate' in param and isinstance(param['validate'], list):
            #    defaultValue = param['validate'][0]['value']
            #self.params[param] = None
    
    def __call__(self, **kwargs):
        dtype_map = {
            "unset": None,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp8_e4m3fn": torch.float8_e4m3fn,
        }

        # ensure only valid params are updated
        args = { key: kwargs[key] for key in kwargs if key in self.params }

        # reload model only if params have changed
        if any(self.params.get(key) != args.get(key) for key in args if key in self.params):
            self.params.update(args)

            if self.params['dtype'] == "Auto":
                self.params['dtype'] = dtype_map[self.params['variant']]
            else:
                self.params['dtype'] = dtype_map[self.params['dtype']]
            
            #self.params['use_safetensors'] = self.params['use_safetensors']
            #self.safety_checker = safety_checker
            #self.params['add_watermarker'] = self.params['add_watermarker']
            self.load_model()

        return self.pipeline
    
    def load_model(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.params['model_id'],
            torch_dtype=self.params['dtype'],
            variant=self.params['variant'],
            use_safetensors=self.params['use_safetensors'],
            safety_checker=None,
            add_watermarker=False,
        )

class Sampling():
    def __init__(self):
        self.params = {
            'positive_prompt': None,
            'negative_prompt': None,
            'num_inference_steps': None,
            'guidance_scale': None,
            'width': None,
            'height': None,
            'pipeline': None,
            'device': None,
        }

        # return images
        self.images = None

        #self.pipe = self.pipe.to("cuda")
        #self.pipe = self.pipe.to("cpu")
    
    def __call__(self, **kwargs):
        if any(self.params.get(key) != kwargs.get(key) for key in kwargs if key in self.params):
            self.params.update(kwargs)

            if not self.params['device']:
                self.params['device'] = device_list[0]['value']

            self.params['pipeline'].to(self.params['device'])
            self.images = self.sample().images

        return self.images

    def sample(self):
        return self.params['pipeline'](
            prompt=self.params['positive_prompt'],
            negative_prompt=self.params['negative_prompt'] if self.params['negative_prompt'] else None,
            num_inference_steps=self.params['num_inference_steps'],
            guidance_scale=self.params['guidance_scale'],
            width=self.params['width'],
            height=self.params['height'],
        )
