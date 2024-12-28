from PIL import Image
import torch
from utils.torch_utils import toTensor, toPIL
from utils.node_utils import NodeBase
from utils.memory_manager import memory_manager

class Preview(NodeBase):
    def execute(self, images, vae):
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            images = images[0]

        if isinstance(images, Image.Image):
            # TODO: support multiple images
            return { 'images_out': images,
                     'width': images.width,
                     'height': images.height }

        if not vae:
            raise ValueError("VAE is required to decode latents")

        device = vae['device']
        model_id = vae['vae'] if 'vae' in vae else vae['model']
        model = self.mm_load(model_id, device)

        latents = 1 / model.config['scaling_factor'] * images
        #latents = self.mm_flash_load(latents, device=device)

        #with torch.no_grad():
        with torch.inference_mode():
            while True:
                try:
                    images = model.decode(latents.to(device, dtype=model.dtype), return_dict=False)[0][0]
                    break
                except torch.OutOfMemoryError as e:
                    if memory_manager.unload_next(device, exclude=model_id):
                        continue
                    else:
                        raise e
                except Exception as e:
                    raise e

        del latents

        images = (images / 2 + 0.5).to('cpu')
        images = toPIL(images)

        return { 'images_out': images,
                 'width': images.width,
                 'height': images.height }


class LoadImage(NodeBase):
    def execute(self, path):
        return { 'images': Image.open(path) }


class SaveImage(NodeBase):
    def execute(self, images: list):
        # save all the images in the list
        for i, image in enumerate(images):
            image.save(f"image_{i}.webp")

        return

class ResizeToDivisible(NodeBase):
    def execute(self, images, divisible_by):
        from PIL.ImageOps import fit
        divisible_by = int(max(1, divisible_by))
        width, height = images.size
        width = width // divisible_by * divisible_by
        height = height // divisible_by * divisible_by
        images = fit(images, (width, height), Image.Resampling.LANCZOS)
        return { 'images_out': images,
                 'width': width,
                 'height': height }

# NOT implemented
class BlendImages(NodeBase):
    def execute(self, source: Image.Image, target: Image.Image, amount: float):
        source = toTensor(source)
        target = toTensor(target)
        blend = source * amount + target * (1 - amount)
        blend = toPIL(blend)

        return { 'blend': blend }
