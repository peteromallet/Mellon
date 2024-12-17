schedulers_config = {
    'FlowMatchEulerDiscreteScheduler': {
        'num_train_timesteps': 1000,
        'shift': 3.0,
        'use_dynamic_shifting': False,
        'base_shift': 0.5,
        'max_shift': 1.15,
        'base_image_seq_len': 256,
        'max_image_seq_len': 4096,
        'invert_sigmas': False,
    },
    'FlowMatchHeunDiscreteScheduler': {
        'num_train_timesteps': 1000,
        'shift': 3.0,
    }
}

vae_config = {
    'SD3': {
        'in_channels': 3,
        'out_channels': 3,
        'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        'block_out_channels': [128, 256, 512, 512],
        'layers_per_block': 2,
        'latent_channels': 16,
    }
}

def dummy_vae(model_id):
    from diffusers import AutoencoderKL
    config = vae_config[model_id]
    return AutoencoderKL(**config)