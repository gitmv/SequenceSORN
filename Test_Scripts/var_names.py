import inspect

def get_default_args(func, exclude=['args', 'kwargs']):
    result = {}
    signature = inspect.signature(func)
    for k, v in signature.parameters.items():
        if k not in exclude:
            if v.default is not inspect.Parameter.empty:
                result[k] = v.default
            else:
                result[k] = ''
    return result

def is_executable():


def func(a,b=3,c='f',*args,**kwargs):
    return #str(a)+str(b)+str(c)+str(args)+str(kwargs)

print(get_default_args(func))
print(type(inspect.getsource(func)))

#print(func.__code__.co_varnames)
#print(func.__code__.co_consts)
#print(func.__kwdefaults__)
#print(func.__defaults__)
