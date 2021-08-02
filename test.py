import time
import numpy as np

start=time.time()
for i in range(1000):
    x=np.random.random(10000)

    y=x>0.5

    x[y]=1.0

print(time.time()-start)


start=time.time()

for i in range(1000):
    x = np.random.random(10000)

    y = x > 0.5

    x[np.where(y)] = 1.0

print(time.time()-start)