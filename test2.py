import os
import torch
import time
from diffusers import ModularPipeline
from diffusers.pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.pipelines.components_manager import ComponentsManager
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_modular import (
    StableDiffusionXLTextEncoderStep,
    StableDiffusionXLDecodeLatentsStep,
    StableDiffusionXLInputStep,
    StableDiffusionXLAutoSetTimestepsStep,
    StableDiffusionXLAutoPrepareLatentsStep,
    StableDiffusionXLAutoPrepareAdditionalConditioningStep,
    StableDiffusionXLAutoDenoiseStep,
)

import logging
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("diffusers").setLevel(logging.INFO)


# define inputs
device = "cuda"
dtype = torch.bfloat16
num_inference_steps = 25
guidance_scale = 7
prompt =  "A beautiful image of a cat"
negative = "lowres, ill, glitches"


# functions for memory info
def reset_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def clear_memory():
    torch.cuda.empty_cache()

def print_mem(mem_size, name):
    mem_gb = mem_size / 1024**3
    mem_mb = mem_size / 1024**2
    print(f"- {name}: {mem_gb:.2f} GB ({mem_mb:.2f} MB)")

def print_memory(message=None):
    """
    Print detailed GPU memory statistics for a specific device.

    Args:
        device_id (int): GPU device ID
    """
    allocated_mem = torch.cuda.memory_allocated(device)
    reserved_mem = torch.cuda.memory_reserved(device)
    mem_on_device = torch.cuda.mem_get_info(device)[0]
    peak_mem = torch.cuda.max_memory_allocated(device)

    print(f"\nGPU:{device} Memory Status {message}:")
    print_mem(allocated_mem, "allocated memory")
    print_mem(reserved_mem, "reserved memory")
    print_mem(peak_mem, "peak memory")
    print_mem(mem_on_device, "mem on device")





# (1) define blocks and nodes(builder)
class StableDiffusionXLMainSteps(SequentialPipelineBlocks):
    block_classes = [
        StableDiffusionXLInputStep,
        StableDiffusionXLAutoSetTimestepsStep,
        StableDiffusionXLAutoPrepareLatentsStep,
        StableDiffusionXLAutoPrepareAdditionalConditioningStep,
        StableDiffusionXLAutoDenoiseStep,
    ]
    block_prefixes = [
        "input",
        "set_timesteps",
        "prepare_latents",
        "prepare_add_cond",
        "denoise",
    ]


text_block = StableDiffusionXLTextEncoderStep()
sdxl_main_block = StableDiffusionXLMainSteps()
decoder_block = StableDiffusionXLDecodeLatentsStep()

text_node = ModularPipeline.from_block(text_block)
sdxl_node = ModularPipeline.from_block(sdxl_main_block)
decoder_node = ModularPipeline.from_block(decoder_block)


# (2) add states to nodes
repo = "stabilityai/stable-diffusion-xl-base-1.0"

components = ComponentsManager()
components.add_from_pretrained(repo, torch_dtype=dtype)


# load components/config into nodes
text_node.update_states(**components.get(["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]))
decoder_node.update_states(vae=components.get("vae"))
sdxl_node.update_states(**components.get(["unet", "scheduler", "vae"]))



# (3) move the components to the device
#text_node.to(device)
#sdxl_node.to(device)
#decoder_node.to(device)

# # alternatively, you can try out the offload strategy
components.enable_auto_cpu_offload(device=device, memory_reserve_margin="18GB")

reset_memory()
# using text_node to generate text embeddings
text_state = text_node(prompt=prompt, negative_prompt=negative)
print_memory("text_node")
print(components)


# warm up
# using sdxl_node to generate images
generator = torch.Generator(device="cuda").manual_seed(0)
latents = sdxl_node(
    **text_state.intermediates,
    generator=generator,
    output="latents"
)
images_output = decoder_node(latents=latents, output="images")
# warm up (done)

print_memory("before generating final image")
print(components)

print("\nGenerating final image...\n")
start = time.time()
generator = torch.Generator(device="cuda").manual_seed(0)
latents = sdxl_node(
    **text_state.intermediates,
    generator=generator,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    width=1024,
    height=1024,
    output="latents"
)
print_memory("after generating final image")
end = time.time()
print(f"\nTime taken: {end - start}\n")
print(components)
