from utils.torch_utils import device_list, default_device

MODULE_MAP = {
    'LoadVAE': {
        'label': 'VAE Loader',
        'description': 'Load the VAE of a Stable Diffusion model',
        'category': 'vae',
        'params': {
            'vae': {
                'label': 'VAE',
                'display': 'output',
                'type': 'vae',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'description': 'The ID of the model to use',
            },
            'device': {
                'label': 'Device',
                'type': 'string',
                'options': device_list,
                'default': default_device,
            },
            'compile': {
                'label': 'Torch Compile',
                'type': 'boolean',
                'default': False,
            },
        },
    },
}