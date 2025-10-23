"""
AnyCapture - A tool to capture local variables from any function

Original Author: luo3300612 (https://github.com/luo3300612)
Original Project: Visualizer (https://github.com/luo3300612/Visualizer)
Current Maintainer: zzaiyan (https://github.com/zzaiyan)

This project is based on the original Visualizer project by luo3300612,
renamed to AnyCapture to avoid conflicts with existing PyPI packages.
"""

import warnings
import types
from bytecode import Bytecode, Instr


class get_local(object):
    cache = {}
    is_activate = False
    max_size = None

    @classmethod
    def is_activated(cls):
        """Return the current activation status"""
        return cls.is_activate

    def __init__(self, *varnames):
        """Initialize with variable names to capture"""
        self.varnames = varnames

    def __call__(self, func):
        # Store the original function's code before any modification
        original_code = func.__code__
        original_globals = func.__globals__
        original_defaults = func.__defaults__
        original_closure = func.__closure__
        
        # Create a true copy of the original function
        true_original_func = types.FunctionType(
            original_code, 
            original_globals, 
            func.__name__, 
            original_defaults, 
            original_closure
        )
        
        c = Bytecode.from_code(func.__code__)

        # store return variable
        extra_code = [Instr('STORE_FAST', '_res')]

        # store local variables
        for var_name in self.varnames:
            extra_code.extend([Instr('LOAD_FAST', var_name),
                              Instr('STORE_FAST', var_name + '_value')])

        # push to TOS
        extra_code.extend([Instr('LOAD_FAST', '_res')])

        for var_name in self.varnames:
            extra_code.extend([Instr('LOAD_FAST', var_name + '_value')])

        extra_code.extend([
            Instr('BUILD_TUPLE', 1 + len(self.varnames)),
            Instr('STORE_FAST', '_result_tuple'),
            Instr('LOAD_FAST', '_result_tuple')
        ])

        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        # callback function
        def wrapper(*args, **kwargs):
            if not type(self).is_activate:
                # If deactivated, call the true original function
                return true_original_func(*args, **kwargs)
            
            res, *values = func(*args, **kwargs)
            for var_idx in range(len(self.varnames)):
                if hasattr(values[var_idx], 'detach'):
                    value = values[var_idx].detach().cpu().numpy()
                else:
                    value = values[var_idx]
                
                cache_key = func.__qualname__ + '.' + self.varnames[var_idx]
                # Initialize cache if not exists
                if cache_key not in type(self).cache:
                    type(self).cache[cache_key] = []
                
                type(self).cache[cache_key].append(value)
                
                # Check queue size if max_size is set and positive
                if (type(self).max_size is not None and 
                    type(self).max_size > 0 and 
                    len(type(self).cache[cache_key]) > type(self).max_size):
                    # Remove the earliest element
                    type(self).cache[cache_key].pop(0)
            return res

        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def get_cache(cls):
        """Return the current cache dictionary"""
        return cls.cache

    @classmethod
    def set_size(cls, max_size):
        """
        Set and normalize the maximum queue capacity
        
        Args:
            max_size (int, optional): Maximum queue capacity. If None or non-positive, 
                                      it will be unlimited capacity.
        
        Returns:
            int or None: Normalized max_size value
        """
        # Check and normalize max_size parameter
        if max_size is None:
            cls.max_size = None
        elif isinstance(max_size, (int, float)):
            # Convert to integer and check if positive
            try:
                max_size_int = int(max_size)
                cls.max_size = max_size_int if max_size_int > 0 else None
            except (ValueError, OverflowError):
                cls.max_size = None
        else:
            # Invalid type, issue warning and set to None
            warnings.warn(
                f"Invalid max_size type: {type(max_size).__name__}. "
                f"Expected int, float, or None. Setting max_size to None (unlimited).",
                UserWarning,
                stacklevel=2
            )
            cls.max_size = None
        
        # If max_size is set and positive, immediately adjust existing cache size
        if cls.max_size is not None and cls.max_size > 0:
            for key in cls.cache:
                while len(cls.cache[key]) > cls.max_size:
                    cls.cache[key].pop(0)
        
        return cls.max_size

    @classmethod
    def activate(cls, max_size=None):
        """
        Activate decorator capture functionality
        
        Args:
            max_size (int, optional): Maximum queue capacity. If None or non-positive,
                                      it will be unlimited capacity. When the recorded
                                      variables are about to exceed the maximum capacity,
                                      the earliest variables will be removed.
        """
        cls.is_activate = True
        cls.set_size(max_size)

    @classmethod
    def deactivate(cls):
        """
        Deactivate decorator capture functionality
        
        After deactivation, decorated functions will behave normally without
        capturing variables. Existing cache data will be preserved.
        """
        cls.is_activate = False
