import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from torchvision.transforms import v2 as tt
import time

model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
dtype = torch.bfloat16
device = 'cuda'
steps = 25
cfg = 7

# Encode the prompt with long prompt support
def get_clip_prompt_embeds(prompt, tokenizer, text_encoder, clip_skip=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
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

    return (prompt_embeds, pooled_prompt_embeds)

def encode_prompt(text_encoder, tokenizer, text_encoder_2, tokenizer_2, prompt="", prompt_2=""):
    prompt = prompt or ""
    prompt_2 = prompt_2 or prompt

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
    
    prompt_embeds, _ = get_clip_prompt_embeds(prompt, tokenizer, text_encoder)
    prompt_embeds_2, pooled_prompt_embeds_2 = get_clip_prompt_embeds(prompt_2, tokenizer_2, text_encoder_2)

    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
    pooled_prompt_embeds_2 = pooled_prompt_embeds_2

    return (prompt_embeds, pooled_prompt_embeds_2)

def vae_decode(latents, model):
    latents = latents.to(dtype=dtype)
    latents = 1 / model.config['scaling_factor'] * latents
    images = model.decode(latents.to(model.device), return_dict=False)[0][0]
    images = images / 2 + 0.5
    images = tt.ToPILImage()(images.to('cpu').clamp(0, 1).float())
    return images

# Load the unet
unet = UNet2DConditionModel.from_pretrained(
    model_id,
    torch_dtype=dtype,
    subfolder="unet",
    variant='fp16',
)

# Load the scheduler
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# Load the text encoders and tokenizers
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")

text_encoder.to(device)
text_encoder_2.to(device)

# Encode the positive prompt
prompt = "A beautiful image of a cat"
prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoder, tokenizer, text_encoder_2, tokenizer_2, prompt)

# Encode the negative prompt
negative_prompt = "glitches, lowres, ill"
negative_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt(text_encoder, tokenizer, text_encoder_2, tokenizer_2, negative_prompt)

# Ensure the prompt and negative prompt have the same length
if prompt_embeds.shape[1] > negative_prompt_embeds.shape[1]:
    negative_prompt_embeds = torch.nn.functional.pad(negative_prompt_embeds, (0, 0, 0, prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]))
elif prompt_embeds.shape[1] < negative_prompt_embeds.shape[1]:
    prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, negative_prompt_embeds.shape[1] - prompt_embeds.shape[1]))

# Load a dummy vae
dummy_vae = AutoencoderKL(
    in_channels=3,
    out_channels=3,
    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    latent_channels=4,
).to(device)

# Random seed
generator = torch.Generator(device=device).manual_seed(0)

# Load the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    unet=unet,
    scheduler=scheduler,
    vae=dummy_vae,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    add_watermarker=False,
).to(device)

print("\nGenerating warm up image...\n")

# Generate the warm up image
images = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    generator=generator,
    guidance_scale=cfg,
    num_inference_steps=steps,
    width=1024,
    height=1024,
    output_type="latent",
).images

#load the vae
vae = AutoencoderKL.from_pretrained(
    model_id, 
    subfolder="vae",
    torch_dtype=dtype,
).to(device)

# Decode the warm up image
images = vae_decode(images, vae)

# All of the above is to warm up the whole pipeline, now we can generate the final image
generator = torch.Generator(device=device).manual_seed(42)
print("\nGenerating final image...\n")
start = time.time()

images = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    generator=generator,
    guidance_scale=cfg,
    num_inference_steps=steps,
    width=1024,
    height=1024,
    output_type="latent",
).images

end = time.time()
print(f"\nTime taken: {end - start}\n")

# Decode the final image
images = vae_decode(images, vae)
#images.save("cat.png")
