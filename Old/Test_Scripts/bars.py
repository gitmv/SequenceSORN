import matplotlib.pyplot as plt
import numpy as np


r=np.random.randint(0,20,2000)
nr=[np.sum(r==i) for i in range(20)]
#print(np.sum(r==0))
print(np.sort(nr))

plt.bar(range(20), np.sort(nr))

#plt.plot(np.sort(nr))

plt.show()

'''
plt.xlabel('clusters')
plt.ylabel('neurons')
plt.suptitle('same input frequency ', size=16)

plt.bar([17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1], [3,4,4,4,4,4,5,5,5,5,5,5,5,6,7,7,22])
print(np.sum([3,4,4,4,4,4,5,5,5,5,5,5,5,6,7,7,22]))
plt.show()

plt.suptitle('different input frequency ', size=16)

plt.bar([20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1], [4,4,4,4,4,4,4,4,5,5,5,5,6,6,6,6,6,6,6,6])
print(np.sum([4,4,4,4,4,4,4,4,5,5,5,5,6,6,6,6,6,6,6,6]))
plt.show()
'''