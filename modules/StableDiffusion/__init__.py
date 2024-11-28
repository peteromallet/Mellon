from utils.torch_utils import device_list, default_device
import torch

MODULE_MAP = {
    'LoadUNet': {
        'label': 'SD UNet loader',
        'description': 'Load the UNet of a Stable Diffusion model',
        'params': {
            'unet': {
                'label': 'UNet',
                'type': 'UNet2DConditionModel',
                'display': 'output',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'description': 'The ID of the model to use',
            },
            'variant': {
                'label': 'Variant',
                'options': ['[unset]', 'fp32', 'fp16', 'bf16', 'fp8_e4m3fn'],
                'postProcess': lambda variant, params: variant if variant != "[unset]" else '',
                'default': 'fp16',
                'description': 'The variant of the checkpoint to use',
            },
            'dtype': {
                'label': 'dtype',
                'options': ['Auto', 'fp32', 'fp16', 'bf16', 'fp8_e4m3fn'],
                'postProcess': lambda dtype, params: {
                    '[unset]': torch.float32,
                    'fp32': torch.float32,
                    'fp16': torch.float16,
                    'bf16': torch.bfloat16,
                    'fp8_e4m3fn': torch.float8_e4m3fn,
                }[dtype if dtype != "Auto" else params['variant']],
                'default': 'Auto',
                'description': 'The data type to convert the model to. If "Auto" is selected, the data type will be inferred from the checkpoint.',
            },
            'use_safetensors': {
                'label': 'Use Safetensors',
                'type': 'boolean',
                'default': True,
                'description': 'Whether to use safetensors file format',
            },
            'device': {
                'label': 'Device',
                'type': 'string',
                'options': device_list,
                'default': default_device,
            },
        },
    },

    'LoadTextEncoder': {
        'label': 'SD CLIPText loader',
        'description': 'CLIPText Model loader helper for Stable Diffusion',
        'params': {
            'clip': {
                'label': 'CLIP Text Encoder',
                'display': 'output',
                'type': 'CLIPTextModel',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'default': 'openai/clip-vit-large-patch14',
                'description': 'The ID of the model to use',
            },
            'device': {
                'label': 'Device',
                'type': 'string',
                'options': device_list,
                'default': default_device,
            },
        },
    },

    'TextEncoder': {
        'label': 'CLIP Text Encoder',
        'description': 'Encodes text into a vector space',
        'params': {
            'clip': {
                'label': 'CLIP model',
                'display': 'input',
                'type': 'CLIPTextModel',
            },
            'prompt': {
                'label': 'Prompt',
                'type': 'string',
                'display': 'textarea',
            },
            'embeds': {
                'label': 'Embeddings',
                'type': 'CLIPTextEmbeddings',
                'display': 'output',
            },
        },
    },

    'SDSampler': {
        'label': 'SD Sampler',
        'description': 'A custom sampler for Stable Diffusion',
        'style': {
            'maxWidth': '320px',
        },
        'params': {
            'unet': {
                'label': 'UNet',
                'display': 'input',
                'type': 'UNet2DConditionModel',
            },
            'positive': {
                'label': 'Positive',
                'display': 'input',
                'type': 'CLIPTextEmbeddings',
            },
            'negative': {
                'label': 'Negative',
                'display': 'input',
                'type': 'CLIPTextEmbeddings',
            },
            'latents_in': {
                'label': 'Latents',
                'display': 'input',
                'type': 'latents',
            },
            'seed': {
                'label': 'Seed',
                'type': 'int',
                'default': 0,
                'min': 0,
                #'max': (1<<53)-1, # max JS integer
            },
            'width': {
                'label': 'W',
                'type': 'int',
                'default': 512,
                'min': 8,
                'max': 8192,
                'step': 8,
                'group': 'dimensions',
            },
            'height': {
                'label': 'H',
                'type': 'int',
                'default': 512,
                'min': 8,
                'max': 8192,
                'step': 8,
                'group': 'dimensions',
            },
            'steps': {
                'label': 'Steps',
                'type': 'int',
                'default': 30,
                'min': 0,
                'max': 10000,
            },
            'cfg': {
                'label': 'Guidance',
                'type': 'float',
                'display': 'slider',
                'default': 7.5,
                'min': 0,
                'max': 20,
            },
            'latents': {
                'label': 'Latents',
                'type': 'latent',
                'display': 'output',
            },
        },
    },
}