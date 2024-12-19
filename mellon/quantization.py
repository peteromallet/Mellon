from config import config
from utils.memory_manager import memory_manager
import os

def set_compile_env():
    if 'CC' in config.environ and config.environ['CC']:
        os.environ['CC'] = config.environ['CC']
    if 'CXX' in config.environ and config.environ['CXX']:
        os.environ['CXX'] = config.environ['CXX']
    if 'TORCH_CUDA_ARCH_LIST' in config.environ and config.environ['TORCH_CUDA_ARCH_LIST']:
        os.environ['TORCH_CUDA_ARCH_LIST'] = config.environ['TORCH_CUDA_ARCH_LIST']

def quanto(model, weights, activations=None, exclude=None, device=None):
    from optimum.quanto import freeze, quantize

    set_compile_env()

    if device:
        model.to(device)

    weights_dtype = f"q{weights.lower()}"
    activations_dtype = f"q{activations.lower()}" if activations != 'none' else None

    weights_module = getattr(__import__('optimum.quanto', fromlist=[weights_dtype]), weights_dtype)
    activations_module = None
    if activations_dtype:
        activations_module = getattr(__import__('optimum.quanto', fromlist=[activations_dtype]), activations_dtype)

    exclude = exclude or []
    if isinstance(exclude, str):
        exclude = [item.strip() for item in exclude.split(',')]

    quantize(model, weights=weights_module, activations=activations_module, exclude=exclude)
    freeze(model)

    return model

def torchao(model, weights, device=None):
    from torchao.quantization.quant_api import quantize_

    set_compile_env()

    weights_dtype = f"{weights.lower()}"
    weights_module = getattr(__import__('torchao.quantization.quant_api', fromlist=[weights_dtype]), weights_dtype)

    quantize_(model, weights_module(), device=device)

class NodeQuantization():
    def quantize(self, type, **kwargs):
        if type == 'none':
            return
        elif type == 'torchao':
            return self._torchao(**kwargs)
        elif type == 'quanto':
            return self._quanto(**kwargs)
        else:
            raise ValueError(f"Invalid quantization type: {type}")

    def _torchao(self, model=None, device=None, torchao_weights=None, torchao_individual_layers=False, **kwargs):
        model_ids = model
        if not isinstance(model_ids, list):
            model_ids = [model_ids]
        
        device = device if torchao_individual_layers else None

        for model_id in model_ids:
            memory_manager.unload_all(exclude=[model_id])

            if not torchao_individual_layers:
                memory_manager.unload_all(exclude=[model_id])
                self.mm_load(model_id, device)
            else:
                memory_manager.unload_all()

            model = torchao(self.mm_get(model_id), torchao_weights, device=device)
            self.mm_update(model_id, model=model)

    def _quanto(self, model=None, device=None, quanto_weights=None, quanto_activations=None, quanto_exclude=None, **kwargs):
        model_ids = model
        if not isinstance(model_ids, list):
            model_ids = [model_ids]

        for model_id in model_ids:
            memory_manager.unload_all(exclude=[model_id])
            self.mm_load(model_id, device)
            model = quanto(self.mm_get(model_id), quanto_weights, activations=quanto_activations, exclude=quanto_exclude)
            self.mm_update(model_id, model=model)

