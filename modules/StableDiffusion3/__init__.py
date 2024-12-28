from utils.torch_utils import device_list, default_device, str_to_dtype
from utils.hf_utils import list_local_models

MODULE_MAP = {
    'SD3UnifiedLoader': {
        'label': 'SD3 Unified Loader',
        'category': 'samplers',
        'params': {
            'model': {
                'label': 'SD3 Pipeline',
                'type': 'SD3Pipeline',
                'display': 'output',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'options': list_local_models(),
                'display': 'autocomplete',
                'no_validation': True,
                'default': 'stabilityai/stable-diffusion-3.5-large',
            },
            'dtype': {
                'label': 'Dtype',
                'type': 'string',
                'options': ['auto', 'float32', 'float16', 'bfloat16'],
                'default': 'bfloat16',
                'postProcess': str_to_dtype,
            },
            'device': {
                'label': 'Device',
                'type': 'string',
                'options': device_list,
                'default': default_device,
            },
            'load_t5': {
                'label': 'Load T5 Encoder',
                'type': 'boolean',
                'default': True,
            },
        },
    },

    'SD3TransformerLoader': {
        'label': 'SD3 Transformer loader',
        'description': 'Load the Transformer of an SD3 model',
        'category': 'samplers',
        'params': {
            'model': {
                'label': 'Transformer',
                'type': 'SD3Transformer2DModel',
                'display': 'output',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'options': list_local_models(),
                'display': 'autocomplete',
                'no_validation': True,
                'default': 'stabilityai/stable-diffusion-3.5-large',
            },
            'dtype': {
                'label': 'dtype',
                'options': ['auto', 'float32', 'float16', 'bfloat16'],
                'default': 'bfloat16',
                'postProcess': str_to_dtype,
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
            'model': {
                'label': 'SD3 Encoders',
                'display': 'output',
                'type': 'SD3TextEncoders',
            },
            'model_id': {
                'label': 'Model ID',
                'type': 'string',
                'options': list_local_models(),
                'display': 'autocomplete',
                'no_validation': True,
                'default': 'stabilityai/stable-diffusion-3.5-large',
            },
            'dtype': {
                'label': 'Dtype',
                'type': 'string',
                'options': ['auto', 'float32', 'float16', 'bfloat16'],
                'default': 'bfloat16',
            },
            'device': {
                'label': 'Device',
                'type': 'string',
                'options': device_list,
                'default': default_device,
            },
            'load_t5': {
                'label': 'Load T5 Encoder',
                'type': 'boolean',
                'default': True,
            },
        },
    },

    'SD3SimplePromptEncoder': {
        'label': 'SD3 Simple Prompt Encoder',
        'description': 'Encode a prompt into embeddings for SD3',
        'category': 'text-encoders',
        'params': {
            'text_encoders': {
                'label': 'SD3 Encoders | SD3 Pipeline',
                'display': 'input',
                'type': ['SD3TextEncoders', 'SD3Pipeline'],
            },
            'embeds': {
                'label': 'Prompt embeds',
                'display': 'output',
                'type': 'SD3Embeddings',
            },
            'prompt': {
                'label': 'Positive Prompt',
                'type': 'string',
                'display': 'textarea',
            },
            'negative_prompt': {
                'label': 'Negative Prompt',
                'type': 'string',
                'display': 'textarea',
            },
        },
    },

    'SD3PromptEncoder': {
        'label': 'SD3 Prompt Encoder',
        'category': 'text-encoders',
        'params': {
            'text_encoders': {
                'label': 'SD3 Encoders | SD3 Pipeline',
                'display': 'input',
                'type': ['SD3TextEncoders', 'SD3Pipeline'],
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
        'category': 'samplers',
        'style': {
            'maxWidth': '320px',
        },
        'params': {
            'transformer': {
                'label': 'Transformer | SD3 Pipeline',
                'display': 'input',
                'type': ['SD3Transformer2DModel', 'SD3Pipeline'],
            },
            'prompt': {
                'label': 'Prompt',
                'display': 'input',
                'type': 'SD3Embeddings',
            },
            'negative_prompt': {
                'label': 'Negative prompt',
                'display': 'input',
                'type': 'SD3Embeddings',
            },
            'latents_in': {
                'label': 'Latents',
                'display': 'input',
                'type': 'latent',
                'onChange': { 'action': 'disable', 'target': { 'connected': ['dimensions_group'], 'disconnected': ['denoise'] } },
            },
            'latents': {
                'label': 'Latents',
                'type': 'latent',
                'display': 'output',
            },
            'width': {
                'label': 'Width',
                'type': 'int',
                'display': 'text',
                'default': 1024,
                'min': 8,
                'max': 8192,
                'step': 8,
                'group': 'dimensions',
            },
            'height': {
                'label': 'Height',
                'type': 'int',
                'display': 'text',
                'default': 1024,
                'min': 8,
                'max': 8192,
                'step': 8,
                'group': 'dimensions',
            },
            'resolution_picker': {
                'label': 'Resolution',
                'display': 'ui',
                'type': 'dropdownIcon',
                'options': [
                    { 'label': ' 720×1280 (9:16)', 'value': [720, 1280] },
                    { 'label': ' 768×1344 (0.57)', 'value': [768, 1344] },
                    { 'label': ' 768×1280 (3:5)', 'value': [768, 1280] },
                    { 'label': ' 832×1152 (3:4)', 'value': [832, 1152] },
                    { 'label': '1024×1024 (1:1)', 'value': [1024, 1024] },
                    { 'label': ' 1152×832 (4:3)', 'value': [1152, 832] },
                    { 'label': ' 1280×768 (5:3)', 'value': [1280, 768] },
                    { 'label': ' 1344×768 (1.75)', 'value': [1344, 768] },
                    { 'label': ' 1280×720 (16:9)', 'value': [1280, 720] },
                ],
                'target': ['width', 'height'],
                'group': 'dimensions',
            },
            'seed': {
                'label': 'Seed',
                'type': 'int',
                'default': 0,
                'min': 0,
                'display': 'random',
            },
            'steps': {
                'label': 'Steps',
                'type': 'int',
                'default': 30,
                'min': 1,
                'max': 1000,
            },
            'cfg': {
                'label': 'Guidance',
                'type': 'float',
                'default': 5,
                'min': 0,
                'max': 100,
            },
            'denoise': {
                'label': 'Denoise',
                'type': 'float',
                'default': 1.0,
                'min': 0,
                'max': 1,
                'display': 'slider',
                'step': 0.05,
            },
            'scheduler': {
                'label': 'Scheduler',
                'display': 'select',
                'type': ['string', 'scheduler'],
                'options': {
                    'FlowMatchEulerDiscreteScheduler': 'Flow Match Euler Discrete',
                    'FlowMatchHeunDiscreteScheduler': 'Flow Match Heun Discrete',
                },
                'default': 'FlowMatchEulerDiscreteScheduler',
            },
            'shift': {
                'label': 'Shift',
                'type': 'float',
                'default': 3.0,
                'min': 0,
                'max': 12,
                'step': 0.05,
                'group': { 'key': 'scheduler', 'label': 'Scheduler options', 'display': 'collapse' },
            },
            'use_dynamic_shifting': {
                'label': 'Use dynamic shifting',
                'type': 'boolean',
                'default': False,
                'group': 'scheduler',
            },
        },
    },
}

quantization_params = {
    'quantization': {
        'label': 'Quantization',
        'options': {
            'none': 'None',
            'quanto': 'Quanto',
            'torchao': 'TorchAO',
        },
        'default': 'none',
        'onChange': 'showGroup',
    },

    # Quanto Quantization
    'quanto_weights': {
        'label': 'Weights',
        'options': ['int2', 'int4', 'int8', 'float8'],
        'default': 'float8',
        'group': { 'key': 'quanto', 'label': 'Quanto Quantization', 'display': 'group', 'hidden': True, 'direction': 'column' },
    },
    'quanto_activations': {
        'label': 'Activations',
        'options': ['none', 'int2', 'int4', 'int8', 'float8'],
        'default': 'none',
        'group': 'quanto'
    },
    'quanto_exclude': {
        'label': 'Exclude blocks',
        'description': 'Comma separated list of block names to exclude from quantization',
        'type': 'string',
        'default': '',
        'group': 'quanto'
    },

    # TorchAO Quantization
    'torchao_weights': {
        'label': 'Weights',
        'options': {
            'int8_weight_only': 'int8 weight',
            'int4_weight_only': 'int4 weight',
            'int8_dynamic_activation_int8_weight': 'int8 weight + activation',
        },
        'default': 'int8_weight_only',
        'group': { 'key': 'torchao', 'label': 'TorchAO Quantization', 'display': 'group', 'hidden': True, 'direction': 'column' },
    },
    'torchao_individual_layers': {
        'label': 'Quantize each layer individually',
        'type': 'boolean',
        'default': False,
        'group': 'torchao'
    },
}

MODULE_MAP['SD3TransformerLoader']['params'].update(quantization_params)
MODULE_MAP['SD3TextEncodersLoader']['params'].update(quantization_params)
