"""
AnyCapture - A tool to capture local variables from any function

Original Author: luo3300612 (https://github.com/luo3300612)
Original Project: Visualizer (https://github.com/luo3300612/Visualizer)
Current Maintainer: zzaiyan (https://github.com/zzaiyan)

This project is based on the original Visualizer project by luo3300612,
renamed to AnyCapture to avoid conflicts with existing PyPI packages.
"""

from bytecode import Bytecode, Instr

class get_local(object):
    cache = {}
    is_activate = False

    def __init__(self, varname):
        self.varname = varname

    def __call__(self, func):
        if not type(self).is_activate:
            return func

        type(self).cache[func.__qualname__] = []
        c = Bytecode.from_code(func.__code__)
        extra_code = [
                         Instr('STORE_FAST', '_res'),
                         Instr('LOAD_FAST', self.varname),
                         Instr('STORE_FAST', '_value'),
                         Instr('LOAD_FAST', '_res'),
                         Instr('LOAD_FAST', '_value'),
                         Instr('BUILD_TUPLE', 2),
                         Instr('STORE_FAST', '_result_tuple'),
                         Instr('LOAD_FAST', '_result_tuple'),
                     ]
        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        def wrapper(*args, **kwargs):
            res, values = func(*args, **kwargs)
            type(self).cache[func.__qualname__].append(values.detach().cpu().numpy())
            return res
        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def activate(cls):
        cls.is_activate = True
