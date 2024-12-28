import torch
from diffusers import AutoencoderKL
from utils.node_utils import NodeBase
from utils.hf_utils import is_local_files_only
from utils.torch_utils import toPIL, toLatent

class LoadVAE(NodeBase):
    #is_compiled = False
    
    def execute(self, model_id, device):
        #if not compile and self.is_compiled:
        #    self.mm_unload(vae)

        vae = AutoencoderKL.from_pretrained(
            model_id, 
            subfolder="vae", 
            local_files_only=is_local_files_only(model_id),
        )

        vae = self.mm_add(vae, priority=2)
        
        """
        if compile:
            # we free up all the GPU memory to perform the intensive compilation
            memory_manager.unload_all(exclude=vae)

            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

            compiled = self.mm_load(vae, device=device).to(memory_format=torch.channels_last)
            compiled.decode = torch.compile(compiled.decode, mode='max-autotune', fullgraph=True)
            self.mm_update(vae, model=compiled)
            del compiled
            memory_flush(deep=True)
            self.is_compiled = True
        """

        return { 'model': { 'vae': vae, 'device': device } }

class VAEEncode(NodeBase):
    def execute(self, model, images, divisible_by):
        device = model['device']
        model_id = model['vae'] if 'vae' in model else model['model']

        if divisible_by > 1:
            from modules.BasicImage.BasicImage import ResizeToDivisible
            images = ResizeToDivisible()(images=images, divisible_by=divisible_by)['images_out']
        
        latents = self.mm_try(
            lambda: self.encode(model_id, images, device),
            device,
            exclude=model_id
        )

        return { 'latents': latents }
    
    def encode(self, model_id, images, device):
        model = self.mm_get(model_id)
        model = model.to(device)
        images = toLatent(images).to(model.device, dtype=model.dtype)

        latents = model.encode(images).latent_dist.sample()
        latents = latents * model.config.scaling_factor
        latents = latents.to('cpu')
        del images, model
        return latents

class VAEDecode(NodeBase):
    def execute(self, model, latents):
        device = model['device']
        model_id = model['vae'] if 'vae' in model else model['model']

        images = self.mm_try(
            lambda: self.decode(model_id, latents, device),
            device,
            exclude=model_id
        )

        return { 'images': images }
    
    def decode(self, model_id, latents, device):
        model = self.mm_get(model_id)
        model = model.to(device)
        latents = 1 / model.config['scaling_factor'] * latents
        images = model.decode(latents.to(model.device, dtype=model.dtype), return_dict=False)[0][0]
        del latents, model
        images = images / 2 + 0.5
        images = toPIL(images.to('cpu'))
        return images
