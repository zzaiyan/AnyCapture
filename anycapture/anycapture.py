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

    def __init__(self, *varnames):
        """varname: tuple"""
        self.varnames = varnames

    def __call__(self, func):
        if not type(self).is_activate:
            return func

        c = Bytecode.from_code(func.__code__)

        # store return variable
        extra_code = [Instr('STORE_FAST', '_res')]

        # store local variables
        for var_name in self.varnames:
            type(self).cache[func.__qualname__ +
                             '.' + var_name] = []  # create cache
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

        # callback
        def wrapper(*args, **kwargs):
            res, *values = func(*args, **kwargs)
            for var_idx in range(len(self.varnames)):
                value = values[var_idx].detach().cpu().numpy()
                type(self).cache[func.__qualname__ + '.' +
                                 self.varnames[var_idx]].append(value)
            return res

        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def activate(cls):
        cls.is_activate = True
