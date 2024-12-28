MODULE_MAP = {
    'Preview': {
        'label': 'Preview Image',
        'description': 'Preview an image',
        'category': 'image',
        'params': {
            'vae': {
                'label': 'VAE',
                'display': 'input',
                'type': ['vae', 'SD3Pipeline'],
                'description': 'VAE to decode latents. Required only if input images are latents.',
            },
            'images': {
                'label': 'Images | Latents',
                'display': 'input',
                'type': ['image', 'latent'],
            },
            'images_out': {
                'label': 'Images',
                'display': 'output',
                'type': 'image',
            },
            'width': {
                'label': 'Width',
                'type': 'int',
                'display': 'output',
            },
            'height': {
                'label': 'Height',
                'type': 'int',
                'display': 'output',
            },
            'preview': {
                'label': 'Preview',
                'display': 'ui',
                'source': 'images_out',
                'type': 'image',
            },
        },
    },

    'SaveImage': {
        'label': 'Save Image',
        'description': 'Save an image',
        'category': 'image',
        'params': {
            'images': {
                'label': 'Images',
                'display': 'input',
                'type': 'image',
            },
        },
    },

    'LoadImage': {
        'label': 'Load Image',
        'description': 'Load an image from path',
        'category': 'image',
        'params': {
            'path': {
                'label': 'Path',
                'type': 'string',
            },
            'images': {
                'label': 'Image',
                'display': 'output',
                'type': 'image',
            },
        },
    },

    'ResizeToDivisible': {
        'label': 'Resize to Divisible',
        'description': 'Resize an image to be divisible by a given number',
        'category': 'image',
        'style': {
            'maxWidth': 300,
        },
        'params': {
            'images': {
                'label': 'Images',
                'display': 'input',
                'type': 'image',
            },
            'images_out': {
                'label': 'Images',
                'display': 'output',
                'type': 'image',
            },
            'width': {
                'label': 'Width',
                'display': 'output',
                'type': 'int',
            },
            'height': {
                'label': 'Height',
                'display': 'output',
                'type': 'int',
            },
            'divisible_by': {
                'label': 'Divisible By',
                'type': 'int',
                'default': 8,
                'min': 1
            },
        },
    },

    'BlendImages': {
        'label': 'Blend Images',
        'description': 'Blend two images',
        'category': 'image',
        'params': {
            'source': {
                'label': 'Source',
                'display': 'input',
                'type': 'image',
            },
            'target': {
                'label': 'Target',
                'display': 'input',
                'type': 'image',
            },
            'amount': {
                'label': 'Amount',
                'type': 'float',
                'display': 'slider',
                'default': 0.5,
                'min': 0,
                'max': 1,
            },
            'blend': {
                'label': 'Blend',
                'display': 'output',
                'type': 'image',
            },
        },
    },
}
