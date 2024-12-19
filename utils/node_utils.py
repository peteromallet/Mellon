from modules import MODULE_MAP
import torch
import time
from utils.memory_manager import memory_flush, memory_manager
import nanoid

def get_module_params(module_name, class_name):
    params = MODULE_MAP[module_name][class_name]['params'] if module_name in MODULE_MAP and class_name in MODULE_MAP[module_name] else {}
    return { p: params[p]['default'] if 'default' in params[p] else None
            for p in params if not 'display' in params[p] or (params[p]['display'] != 'output' and params[p]['display'] != 'ui') }

def get_module_output(module_name, class_name):
    params = MODULE_MAP[module_name][class_name]['params'] if module_name in MODULE_MAP and class_name in MODULE_MAP[module_name] else {}
    return { p: None for p in params if 'display' in params[p] and params[p]['display'] == 'output' }

def filter_params(params, args):
    return { key: args[key] for key in args if key in params }

def has_changed(params, args):
    return any(params.get(key) != args.get(key) for key in args if key in params)

def are_different(a, b):
    # check if the types are different
    if type(a) != type(b):
        return True
    
    # check if the lengths are different
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return True
        
        for x, y in zip(a, b):
            return are_different(x, y)

    if isinstance(a, dict):
        if a.keys() != b.keys():
            return True

        for key in a:
            return are_different(a[key], b[key])

    if hasattr(a, 'dtype'):
        if not hasattr(b, 'dtype') or a.dtype != b.dtype:
            return True

    if hasattr(a, 'shape'):
        if not hasattr(b, 'shape') or not torch.equal(a, b):
            return True
    
    if hasattr(a, '__dict__') and hasattr(b, '__dict__'):
        return are_different(a.__dict__, b.__dict__)
    
    if a != b:
        return True

    return False

class NodeBase():
    CALLBACK = 'execute'

    def __init__(self, node_id=None):
        self.node_id = node_id
        self.module_name = self.__class__.__module__.split('.')[-1]
        if 'custom.' in self.__class__.__module__:
            self.module_name = self.module_name + '.custom'
        self.class_name = self.__class__.__name__
        self.params = {}
        self.output = get_module_output(self.module_name, self.class_name)
        
        self._mm_model_ids = []
        self._execution_time = 0

    def __call__(self, **kwargs):
        # if the node_id is not set, the class was called directly and not from a node
        # so we call the execute method directly
        if not self.node_id:
            return getattr(self, self.CALLBACK)(**kwargs)

        values = self._validate_params(kwargs)

        execution_time = time.time()

        if self._has_changed(values):
            self.params.update(values)

            # delete previously loaded models
            # TODO: delete a model only if something changed about it
            if self._mm_model_ids:
                memory_manager.delete_model(self._mm_model_ids)
                self._mm_model_ids = []

            try:
                output = getattr(self, self.CALLBACK)(**self.params)
            except Exception as e:
                self.params = {}
                self.output = get_module_output(self.module_name, self.class_name)
                memory_flush(gc_collect=True)
                raise e

            if isinstance(output, dict):
                # Overwrite output values only for existing keys
                #self.output.update({k: output[k] for k in self.output if k in output})
                self.output = output
            else:
                # If only a single value is returned, assign it to the first output
                first_key = next(iter(self.output))
                self.output[first_key] = output

        self._execution_time = time.time() - execution_time

        # for good measure, flush the memory
        memory_flush()

        return self.output

    def __del__(self):
        del self.params, self.output # TODO: check if this actually works with cuda

        if self._mm_model_ids:
            memory_manager.delete_model(self._mm_model_ids)

        memory_flush(gc_collect=True)

    def _validate_params(self, values):
        # get the parameters schema for the module/class
        schema = MODULE_MAP[self.module_name][self.class_name]['params'] if self.module_name in MODULE_MAP and self.class_name in MODULE_MAP[self.module_name] else {}

        # get the default values for the parameters
        defaults = get_module_params(self.module_name, self.class_name)

        # filter out any input args that are not valid parameters
        values = { key: values[key] for key in values if key in defaults }

        # ensure the values are of the correct type
        for key in values:
            if 'type' in schema[key]:
                # type can be a list, used to allow multiple types with input handles (a helper for the UI)
                # the main type is the first one in the list
                type = (schema[key]['type'][0] if isinstance(schema[key]['type'], list) else schema[key]['type']).lower()

                if type.startswith('int'):
                    values[key] = int(values[key])
                elif type == 'float':
                    values[key] = float(values[key])
                elif type.startswith('bool'):
                    values[key] = bool(values[key])
                elif type.startswith('str'):
                    values[key] = str(values[key]) if values[key] is not None else ''

        # we perform a second pass for cross parameter validation when calling the postProcess function
        for key in values:
            # ensure the value is a valid option (mostly for dropdowns)
            if 'options' in schema[key] and not ('no_validation' in schema[key] and schema[key]['no_validation']):
                options = schema[key]['options']

                # options can be in the format: [ 1, 2, 3 ] or { '1': { }, '2': { }, '3': { } }
                if isinstance(options, list):
                    val = [values[key]] if isinstance(values[key], str) else values[key]
                    if any(v not in options for v in val):
                        raise ValueError(f"Invalid value for {key}: {values[key]}")
                elif isinstance(options, dict):
                    val = [values[key]] if isinstance(values[key], str) else values[key]
                    if any(v not in options for v in val):
                        raise ValueError(f"Invalid value for {key}: {values[key]}")
                else:
                    raise ValueError(f"Invalid options for {key}: {options}")

            # call the postProcess function if present
            if 'postProcess' in schema[key]:
                # we pass the value and the entire dict for cross parameter validation
                values[key] = schema[key]['postProcess'](values[key], values)

        # update the default values with the validated values
        defaults.update(values)

        return defaults

    def _has_changed(self, values):
        return any(
            key not in self.params or
            are_different(self.params.get(key), values.get(key))
            for key in values
        )
    
    def mm_add(self, model, model_id=None, device=None, priority=2):
        # if the node_id is not set, the class was called directly and we skip the memory manager
        # it's up to the caller to manage the model
        if not self.node_id:
            return model

        if memory_manager.is_cached(model_id):
            self.mm_update(model_id, model=model, priority=priority)
            return model_id

        model_id = f'{self.node_id}.{model_id}' if model_id else f'{self.node_id}.{nanoid.generate(size=8)}'
        device = device if device else str(model.device)

        self._mm_model_ids.append(model_id)
        return memory_manager.add_model(model, model_id, device=device, priority=priority)

    def mm_get(self, model_id):
        return memory_manager.get_model(model_id) if model_id else None

    def mm_load(self, model_id, device):
        return memory_manager.load_model(model_id, device) if model_id else None

    def mm_unload(self, model_id):
        return memory_manager.unload_model(model_id) if model_id else None

    def mm_update(self, model_id, model=None, priority=None, unload=True):
        return memory_manager.update_model(model_id, model=model, priority=priority, unload=unload) if model_id else None
    
    def mm_flash_load(self, model, model_id=None, device='cpu', priority=3):
        model_id = f'{self.node_id}.{model_id}' if model_id else f'{self.node_id}.{nanoid.generate(size=8)}'
        device = device if device else str(model.device)

        return memory_manager.flash_load(model, model_id, device=device, priority=priority)
