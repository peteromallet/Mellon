import torch

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


def get_clip_prompt_embeds(prompt, tokenizer, text_encoder, num_images_per_prompt = 1, clip_skip=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    max_length = tokenizer.model_max_length - 2 # we are adding special tokens at the beginning and end later
    bos = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(text_encoder.device)
    eos = torch.tensor([tokenizer.eos_token_id]).unsqueeze(0).to(text_encoder.device)
    pad = tokenizer.pad_token_id

    text_input_ids = tokenizer(prompt, truncation = False, return_tensors="pt").input_ids.to(text_encoder.device)
    # remove start and end tokens
    text_input_ids = text_input_ids[:, 1:-1]

    chunks = text_input_ids.split(max_length, dim=-1)

    concat_embeds = []
    pooled_prompt_embeds = None
    for chunk in chunks:
        # add start and end tokens to each chunk
        chunk = torch.cat([bos, chunk, eos], dim=-1)

        # pad the chunk to the max length
        if chunk.shape[-1] < max_length:
            chunk = torch.nn.functional.pad(chunk, (0, max_length - chunk.shape[-1]), value=pad)

        # encode the tokenized text
        prompt_embeds = text_encoder(chunk, output_hidden_states=True)
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        concat_embeds.append(prompt_embeds)

    prompt_embeds = torch.cat(concat_embeds, dim=1)

    #_, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    #prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    #prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    #pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    #pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds, pooled_prompt_embeds


def get_t5_prompt_embeds(prompt, tokenizer, text_encoder, num_images_per_prompt = 1, max_sequence_length=256):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    # could be tokenizer.model_max_length but we are using a more conservative value (256)
    max_length = max_sequence_length - 1  # we only need space for EOS token, T5 does not use BOS token
    eos = torch.tensor([1]).unsqueeze(0).to(text_encoder.device) # </s> is the EOS token
    pad = 0 # pad token is 0

    text_inputs_ids = tokenizer(prompt, truncation = False, add_special_tokens=True, return_tensors="pt").input_ids.to(text_encoder.device)
    # remove end token
    text_inputs_ids = text_inputs_ids[:, :-1]

    chunks = text_inputs_ids.split(max_length, dim=-1)

    concat_embeds = []
    for chunk in chunks:
        # add end token to each chunk
        chunk = torch.cat([chunk, eos], dim=-1)

        # pad the chunk to the max length
        if chunk.shape[-1] < max_length:
            chunk = torch.nn.functional.pad(chunk, (0, max_length - chunk.shape[-1]), value=pad)

        # encode the tokenized text
        prompt_embeds = text_encoder(chunk)[0]
        concat_embeds.append(prompt_embeds)

    prompt_embeds = torch.cat(concat_embeds, dim=1)

    #_, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    #prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    #prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds
