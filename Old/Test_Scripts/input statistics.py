from Old.Grammar.Behaviours_in_use import *

ng = NeuronGroup(1, behaviour={}, net=None)

tg = Text_Generator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.'], set_network_size_to_alphabet_size=True)
tg.set_variables(ng)

x = list(range(200))
y = []
c = []

for i in x:
    tg.new_iteration(ng)
    y.append(tg.char_weighting[ng.current_char_index])
    c.append(ng.current_char)

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.xticks(x,c)
plt.show()