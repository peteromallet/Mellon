from utils.torch_utils import device_list, default_device
import torch

def to_dtype(dtype, params):
    return {
        'auto': None,
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float8_e4m3fn': torch.float8_e4m3fn,
    }[dtype]

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
                'options': ['auto', 'float32', 'float16', 'bfloat16', 'float8_e4m3fn'],
                'default': 'bfloat16',
                'postProcess': to_dtype,
            },
            'device': {
                'label': 'Device',
                'type': 'string',
                'options': device_list,
                'default': default_device,
            },
        },
    },

    'SD3TextEncodersLoader': {
        'label': 'SD3 Text Encoders Loader',
        'description': 'Load both the CLIP and T5 Text Encoders',
        'category': 'text-encoders',
        'params': {
            'text_encoders': {
                'label': 'CLIP+T5 Encoders',
                'display': 'output',
                'type': 'SD3TextEncoders',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
            },
            'dtype': {
                'label': 'Dtype',
                'type': 'string',
                'options': ['auto', 'float32', 'float16', 'bfloat16', 'float8_e4m3fn'],
                'default': 'bfloat16',
            },
            'load_t5': {
                'label': 'Load T5 Encoder',
                'type': 'boolean',
                'default': True,
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
                'label': 'CLIP or CLIP+T5 Encoders',
                'display': 'input',
                'type': ['CLIPTextEncoders', 'SD3TextEncoders'],
                #'onConnect': { 'action': 'disable', 'params': ['t5_encoders', 'text_encoders.type', 'SD3TextEncoders'] },
            },
            't5_encoders': {
                'label': 'T5 Encoders',
                'display': 'input',
                'type': 'T5Encoders',
            },
            'embeds': {
                'label': 'Embeddings',
                'display': 'output',
                'type': 'SD3Embeddings',
            },
            'prompt': {
                'label': 'Prompt',
                'type': 'string',
                'display': 'textarea',
            },
            'prompt_2': {
                'label': 'Prompt CLIP G',
                'type': 'string',
                'display': 'textarea',
                'group': { 'key': 'extra_prompts', 'label': 'Extra Prompts', 'display': 'collapse' },
            },
            'prompt_3': {
                'label': 'Prompt T5',
                'type': 'string',
                'display': 'textarea',
                'group': { 'key': 'extra_prompts', 'label': 'Extra Prompts', 'display': 'collapse' },
            },
            'prompt_scale': {
                'label': 'CLIP L',
                'type': 'float',
                'display': 'slider',
                'default': 1.0,
                'min': 0,
                'max': 2,
                'step': 0.05,
                'group': { 'key': 'prompts_scale', 'label': 'Prompts Scale', 'display': 'collapse' },
            },
            'prompt_scale_2': {
                'label': 'CLIP G',
                'type': 'float',
                'default': 1.0,
                'display': 'slider',
                'min': 0,
                'max': 2,
                'step': 0.05,
                'group': { 'key': 'prompts_scale', 'label': 'Prompts Scale', 'display': 'collapse' },
            },
            'prompt_scale_3': {
                'label': 'T5',
                'type': 'float',
                'display': 'slider',
                'default': 1.0,
                'min': 0,
                'max': 2,
                'step': 0.05,
                'group': { 'key': 'prompts_scale', 'label': 'Prompts Scale', 'display': 'collapse' },
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
                'default': 5,
                'min': 0,
                'max': 100,
            },
            'latents': {
                'label': 'Latents',
                'type': 'latent',
                'display': 'output',
            },
        },
    },
}
