from utils.torch_utils import device_list, default_device
from utils.hf_utils import list_local_models
MODULE_MAP = {
    'LoadVAE': {
        'label': 'VAE Loader',
        'description': 'Load the VAE of a Stable Diffusion model',
        'category': 'vae',
        'params': {
            'model': {
                'label': 'VAE',
                'display': 'output',
                'type': 'vae',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'options': list_local_models(),
                'display': 'autocomplete',
                'no_validation': True,
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