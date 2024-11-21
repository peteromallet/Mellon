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
    
    'LoadT5Encoder': {
        'label': 'T5 Encoder loader',
        'description': 'T5 Encoder loader helper for Stable Diffusion',
        'params': {
            't5': {
                'label': 'T5 Encoder',
                'display': 'output',
                'type': 'T5EncoderModel',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'default': 'stabilityai/stable-diffusion-3.5-large',
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
}