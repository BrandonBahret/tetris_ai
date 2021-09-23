
class InterfaceOmission(BaseException):
    def __init__(self, obj, interface, missing_methods):
        super().__init__()
        self.class_name = obj.__qualname__
        self.iname = interface.__qualname__
        self.methods = [self.method_str(m) for m in missing_methods]

    def method_str(self, method):
        arg_count = method.__code__.co_argcount
        arg_names = method.__code__.co_varnames[:arg_count]
        args = str.join(', ', arg_names)
        name = method.__name__
        function_sig = "{0}({args})".format(name, args=args)
        return function_sig

    def __str__(self):
        msg = "class: `{0}` must implement {1} to satisfy `{2}`"
        return msg.format(self.class_name, self.methods, self.iname)

class InterfaceChecker(type):
    "Validates the presence of required attributes"
    def __new__(cls, name, bases, dct):
        obj = type(name, bases, dct)

        if "__implements__" in dct:
            InterfaceChecker.check(obj, dct["__implements__"])

        return obj

    @staticmethod
    def check_method_signature(m1, m2):
        m1_arg_count = m1.__code__.co_argcount
        m1_arg_names = m1.__code__.co_varnames[:m1_arg_count]

        m2_arg_count = m2.__code__.co_argcount
        m2_arg_names = m2.__code__.co_varnames[:m2_arg_count]
        return m1_arg_names == m2_arg_names

    @staticmethod
    def check_method_signatures(obj, interface):
        default_methods = set(dir(type('',(),{})))
        required = set(dir(interface)) - default_methods

        invalid_method_signatures = []
        for method in required:
            obj_method = obj.__dict__[method]
            inter_method = interface.__dict__[method]
            
            valid_sig = InterfaceChecker.check_method_signature(obj_method, inter_method)
            if valid_sig is False:
                invalid_method_signatures += [inter_method]

        if len(invalid_method_signatures) != 0:
            raise InterfaceOmission(obj, interface, invalid_method_signatures)
            
    @staticmethod
    def check(obj, implements):
        defined = set(dir(obj))
        for interface in implements:
            required = set(dir(interface))
            if not required.issubset(defined):
                methods = list(required - defined)
                methods = [interface.__dict__[m] for m in methods]
                raise InterfaceOmission(obj, interface, methods)

            InterfaceChecker.check_method_signatures(obj, interface)
            