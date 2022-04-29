import numpy as np
import time


s=np.random.rand(1000,1000)
s=s/np.sum(s)
print(np.sum(s))
start=time.time()
for i in range(100):
    add = (np.random.rand(1000,1000)<0.02).astype(np.float64)
    s += add
    s = s / np.sum(s)

print(np.sum(s))
print(time.time()-start)



s = np.random.rand(1000, 1000)
s = s / np.sum(s)
print(np.sum(s))
start = time.time()
for i in range(100):
    add = (np.random.rand(1000, 1000) < 0.02).astype(np.float64)
    mask = add==0
    add[mask] = -np.sum(add)/(np.sum(mask))
    s += add

print(np.sum(s))
print(time.time() - start)
