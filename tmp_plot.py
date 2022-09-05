import matplotlib.pyplot as plt
import numpy as np



class cl:

    def a(self, x,y):
        print(x,y)

    def b(self, z, w, *args, **kwargs):
        self.a(*args, **kwargs, x='f')
        print(z, w)

    def set_functions(self):

        def abc(self, eee):
            print(eee)

        self.abc = abc

g = cl()
g.set_functions()

g.b(y='y', z='z', w='w')
g.abc('sdfasdf')

'''

x=[]
t=1
for i in range(30):
   x.append(t)
   t=t-t*0.2+(1-t)*0.05

for i in range(60):
   x.append(t)
   t=t+(1-t)*0.05

plt.plot(x)


x=[]
t=0
for i in range(30):
   x.append(1-t)
   t=np.clip(t+0.173,0,1)

for i in range(60):
   x.append(1-t)
   t=np.clip(t-0.173,0,1)

plt.plot(x)


plt.show()
'''

