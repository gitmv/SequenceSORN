from time import sleep
from dask import *
import numpy as np
import time

'''

def inc(x):
    sleep(1)
    return x + 1

def add(x, y):
    sleep(1)
    return x + y

x = delayed(inc)(1)
y = delayed(inc)(2)
z = delayed(add)(x, y)

#print(z)

start = time.time()

print(z.compute())

print(time.time()-start)

'''

class test:

    def __init__(self):
        self.x = np.zeros(100000)+100
        self.a = np.zeros(100000)+100
        self.b = np.zeros(100000)+100
        self.c = np.zeros(100000)+100
        self.acc = np.zeros(100000)

    def ai(self):
        sleep(1)
        self.a += self.x
        self.acc += self.a

    def bi(self):
        sleep(1)
        self.b += self.x
        self.acc += self.b

    def ci(self):
        sleep(1)
        self.c += self.x
        self.acc += self.c

    def res(self):
        return self.a+self.b+self.c


t = test()

parallel_cmds=[]

parallel_cmds.append(delayed(t.ai)())
parallel_cmds.append(delayed(t.bi)())
parallel_cmds.append(delayed(t.ci)())
parallel_cmds.append(delayed(t.res)())

for i in range(100):
    start = time.time()
    results = compute(*parallel_cmds)
    print(time.time()-start)
    print(t.acc)


#dask does not use multiple cores by default!
