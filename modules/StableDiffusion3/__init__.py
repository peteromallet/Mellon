from utils.torch_utils import device_list, default_device
import torch

MODULE_MAP = {
    'LoadSD3Transformer': {
        'label': 'SD3 Transformer loader',
        'description': 'Load the Transformer of a Stable Diffusion model',
        'params': {
            'transformer': {
                'label': 'Transformer',
                'type': 'SD3Transformer2DModel',
                'display': 'output',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'default': 'stabilityai/stable-diffusion-3.5-large',
                'description': 'The ID of the model to use',
            },
            'dtype': {
                'label': 'dtype',
                'options': ['[unset]', 'fp32', 'fp16', 'bf16', 'fp8_e4m3fn'],
                'postProcess': lambda dtype, params: {
                    '[unset]': '',
                    'fp32': torch.float32,
                    'fp16': torch.float16,
                    'bf16': torch.bfloat16,
                    'fp8_e4m3fn': torch.float8_e4m3fn,
                }[dtype],
                'default': 'bf16',
            },
            'device': {
                'label': 'Device',
                'type': 'string',
                'options': device_list,
                'default': default_device,
            },
        },
    },
    
    'SD3TextEncoder': {
        'label': 'SD3 Text Encoder',
        'description': 'Encode text for SD3',
        'params': {
            'text_encoders': {
                'label': 'CLIP Encoders',
                'display': 'input',
                'type': 'CLIPTextEncoders',
            },
            't5_encoders': {
                'label': 'T5 Encoders',
                'display': 'input',
                'type': 'T5Encoders',
            },
            'prompt': {
                'label': 'Prompt CLIP L',
                'type': 'string',
                'display': 'textarea',
            },
            'prompt_2': {
                'label': 'Prompt CLIP G',
                'type': 'string',
                'display': 'textarea',
            },
            'prompt_3': {
                'label': 'Prompt T5',
                'type': 'string',
                'display': 'textarea',
            },
            'embeds': {
                'label': 'Embeddings',
                'display': 'output',
                'type': 'SD3Embeddings',
            },
        },
    },


    'SD3Sampler': {
        'label': 'SD3 Sampler',
        'description': 'A custom sampler for Stable Diffusion 3',
        'style': {
            'maxWidth': '320px',
        },
        'params': {
            'transformer': {
                'label': 'Transformer',
                'display': 'input',
                'type': 'SD3Transformer2DModel',
            },
            'positive': {
                'label': 'Positive',
                'display': 'input',
                'type': 'SD3Embeddings',
            },
            'negative': {
                'label': 'Negative',
                'display': 'input',
                'type': 'SD3Embeddings',
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
                'default': 1024,
                'min': 8,
                'max': 8192,
                'step': 8,
                'group': 'dimensions',
            },
            'height': {
                'label': 'H',
                'type': 'int',
                'default': 1024,
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
