import inspect

class MyClass():

    def __init__(self, x):
        mod = inspect.getmodule(inspect.stack()[1][0])
        print(len(mod.__dict__))
        print(locals())

        for k,v in mod.__dict__.items():
            if k not in locals():
                locals()[k]=v

        print(eval(x))