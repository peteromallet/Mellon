from PIL import Image
import torch
from utils.torch_utils import toTensor, toPIL
from utils.node_utils import NodeBase

class Preview(NodeBase):
    """
    def __call__(self, **kwargs):
        values = self._validate_params(kwargs)
        images = values.get('images', None)

        self.images = images[0] if isinstance(images, list) else images
    """
    def execute(self, images, vae):
        if isinstance(images, Image.Image):
            # TODO: support multiple images
            self.output['images'] = images[0] if isinstance(images, list) else images
            return
        
        if not vae:
            raise ValueError("VAE is required to decode latents")

        latents = 1 / vae['model'].config['scaling_factor'] * images

        #with torch.no_grad():
        with torch.inference_mode():
            images = vae['model'].to(vae['device']) \
                .decode(latents.to(vae['device'], dtype=vae['model'].dtype), return_dict=False)[0][0]

        del latents
        vae['model'] = vae['model'].to('cpu')

        images = (images / 2 + 0.5).clamp(0, 1).to('cpu')
        images = toPIL(images)

        self.output['images'] = images


class LoadImage(NodeBase):
    def execute(self, path):
        self.output['images'] = Image.open(path)



class SaveImage(NodeBase):
    def execute(self, images: list):
        # save all the images in the list
        for i, image in enumerate(images):
            image.save(f"image_{i}.webp")


# NOT implemented
class BlendImages(NodeBase):
    def execute(self, source: Image.Image, target: Image.Image, amount: float):
        source = toTensor(source)
        target = toTensor(target)
        blend = source * amount + target * (1 - amount)
        blend = toPIL(blend)

        self.output['blend'] = blend
