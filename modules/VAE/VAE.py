import torch
from diffusers import AutoencoderKL
from utils.node_utils import NodeBase
from utils.memory_manager import memory_flush, memory_manager

class LoadVAE(NodeBase):
    is_compiled = False
    
    def execute(self, model_id, device, compile):
        if not compile and self.is_compiled:
            self.mm_unload(vae)


        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        vae = self.mm_add(vae, priority=2)
        
        if compile:
            # we free up all the GPU memory to perform the intensive compilation
            memory_manager.unload_all(exclude=vae)

            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

            compiled = self.mm_load(vae, device=device).to(memory_format=torch.channels_last)
            compiled.decode = torch.compile(compiled.decode, mode='max-autotune', fullgraph=True)
            self.mm_update(vae, model=compiled)
            del compiled
            memory_flush(deep=True)
            self.is_compiled = True

        return { 'vae': { 'model': vae, 'device': device } }
