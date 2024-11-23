from utils.torch_utils import device_list, default_device
import torch

MODULE_MAP = {
    'T5EncoderModelLoader': {
        'label': 'Load T5 Encoder',
        'description': 'Load T5 Encoder',
        'category': 't5',
        'params': {
            't5_encoders': {
                'label': 'T5 Encoders',
                'display': 'output',
                'type': 'T5Encoders',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
            },
            'dtype': {
                'label': 'Dtype',
                'type': 'string',
                'options': {'auto': 'Auto', 'float32': 'fp32', 'float16': 'fp16', 'bfloat16': 'bf16', 'float8_e4m3fn': 'fp8_e4m3fn'},
                'default': 'float16',
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