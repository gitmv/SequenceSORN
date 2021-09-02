import matplotlib.pyplot as plt

#add text
plt.text(0.01, 0.8, r'$\sum(a*b^d)$',fontsize=50)#\alpha > \beta

#hide axes
fig = plt.gca()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
#plt.draw() #or savefig
plt.show()