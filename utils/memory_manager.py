import torch
import gc
import time
from utils.torch_utils import device_list
from enum import Enum
import logging
logger = logging.getLogger('mellon')


def memory_flush(gc_collect=False, deep=False):
    if gc_collect or deep:
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        #torch.cuda.synchronize()
        torch.cuda.ipc_collect()

        if deep:
            for _, d in device_list.items():
                torch.cuda.reset_max_memory_allocated(d['index'])
                torch.cuda.reset_peak_memory_stats(d['index'])

class Priority(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class MemoryManager:
    def __init__(self, memory_threshold=.9):
        self.cache = {}
        self.memory_threshold = memory_threshold

    def add_model(self, model, model_id, device='cpu', priority=Priority.MEDIUM):
        if isinstance(priority, str):
            priority = priority.upper()
            priority = Priority[priority] if priority in Priority else Priority.MEDIUM

        if model_id not in self.cache:
            self.cache[model_id] = {
                'model': model,
                'device': device,           # device the model is currently on
                'priority': priority,       # priority, lower priority models are unloaded first
                'last_used': time.time(),   # time the model was last used
            }

        return model_id

    def get_available_memory(self, device):
        return torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)

    def get_model(self, model_id):
        return self.cache[model_id]['model'] if model_id in self.cache else None

    def load_model(self, model_id, device):
        self.cache[model_id]['last_used'] = time.time()
        x = self.cache[model_id]['model']

        if device == str(x.device):
            return x

        if device == 'cpu':
            return self.unload_model(model_id)

        device_index = device_list[device]['index']

        cache_priority = []
        # Sort models by priority and last_used
        for id, model in self.cache.items():
            if model['device'] == device:
                cache_priority.append((model['priority'].value, model['last_used'], id))

        cache_priority.sort()
        memory_flush()

        while True:
            # Attempt to load the model
            try:
                x = x.to(device)
                self.cache[model_id].update({
                    'model': x,
                    'device': device
                })
                return x

            except RuntimeError as e:
                if "out of memory" not in str(e):
                    raise e

                if not cache_priority:
                    logger.debug("No more models to unload, cannot free sufficient memory")
                    raise e

                next_model_id = cache_priority.pop(0)[2]
                logger.debug(f"OOM error, unloading lower priority model: {next_model_id}")
                self.unload_model(next_model_id)


    def unload_model(self, model_id):
        if model_id in self.cache:
            self.cache[model_id]['model'] = self.cache[model_id]['model'].to('cpu')
            self.cache[model_id]['device'] = 'cpu'
            memory_flush()

        return self.cache[model_id]['model']
    
    def unload_all(self, exclude=[]):
        if not isinstance(exclude, list):
            exclude = [exclude]

        for model_id in self.cache:
            if model_id not in exclude:
                self.unload_model(model_id)

    def delete_model(self, model_id):
        model_id = model_id if isinstance(model_id, list) else [model_id]

        for m in model_id:
            if m in self.cache:
                del self.cache[m]['model']
                del self.cache[m]

        memory_flush()

    def update_model(self, model_id, model=None, priority=None):
        if model_id in self.cache:
            if model:
                self.cache[model_id]['model'] = model
                memory_flush()
            if priority:
                self.cache[model_id]['priority'] = priority



memory_manager = MemoryManager()
