from modules import MODULE_MAP
import torch
import time

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

def compare_values(a, b):
    # if both are tensors
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return not torch.equal(a, b)
    
    # if only one is a tensor, they are certainly different
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return True
    
    # TODO: probably need to add support for other types

    return a != b

class NodeBase():
    CALLBACK = 'execute'

    def __init__(self):
        self.module_name = self.__class__.__module__.split('.')[-1]
        self.class_name = self.__class__.__name__
        self.params = {}
        self.output = get_module_output(self.module_name, self.class_name)

        self._execution_time = 0

    def __call__(self, **kwargs):
        values = self._validate_params(kwargs)

        execution_time = time.time()

        if self._has_changed(values):
            self.params.update(values)
            
            getattr(self, self.CALLBACK)(**self.params)

        self._execution_time = time.time() - execution_time

        return self.output

    def __del__(self):
        del self.params, self.output

        # TODO: this is very aggressive because it will clean the cache even if the node doesn't use pytorch or cuda
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                    values[key] = str(values[key])

        # we perform a second pass to for cross parameter validation when calling the postProcess function
        for key in values:
            # ensure the value is a valid option (mostly for dropdowns)
            if 'options' in schema[key]:
                options = schema[key]['options']

                # options can be in the format: [ 1, 2, 3 ] or { '1': { }, '2': { }, '3': { } }
                if isinstance(options, list):
                    if values[key] not in options:
                        raise ValueError(f"Invalid value for {key}: {values[key]}")
                elif isinstance(options, dict):
                    if not options[values[key]]:
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
            compare_values(self.params.get(key), values.get(key))
            for key in values
        )
