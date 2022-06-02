import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
#people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(20)
performance = np.ones(20)*0.87
#performance[19]=4.35

ax.barh(y_pos, performance, align='center')
#ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
#ax.set_xlabel('Performance')
#ax.set_title('How fast do you want to go today?')

plt.show()