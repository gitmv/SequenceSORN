from Old.Grammar.Behaviors_in_use import *

ng = NeuronGroup(1, behavior={}, net=None)

tg = TextGenerator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.'], set_network_size_to_alphabet_size=True)
tg.initialize(ng)

x = list(range(200))
y = []
c = []

for i in x:
    tg.iteration(ng)
    y.append(tg.char_weighting[ng.current_char_index])
    c.append(ng.current_char)

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.xticks(x,c)
plt.show()