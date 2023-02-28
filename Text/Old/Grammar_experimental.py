####################################

# State transition table
REBER_TRANSITIONS = [
    [('T', 1), ('P', 2)],  # 0=B
    [('X', 3), ('S', 1)],  # 1=BT
    [('V', 4), ('T', 2)],  # 2=BP
    [('X', 2), ('S', 5)],  # 3=BTX
    [('P', 3), ('V', 5)],  # 4=BPV
    [('E', -1)],  # 5=BTXS
]

def make_reber():
    idx = 0
    out = 'B'
    while idx != -1:
        ts = REBER_TRANSITIONS[idx]
        symbol, idx = random.choice(ts)
        out += symbol
    return out

def get_reber_text(n_blocks=100):
    return [make_reber()+' ' for i in range(n_blocks)]


def calculate_parameters(h):
    b = 0.01/h+0.22 #1
    c = 0.4/h+3.6   #40
    th = np.tanh(c*h) #/100.0
    return b, c, th

def calculate_parameters(input_density, text, add_to_genome):
    h = 1.0 / len(''.join(text)) * 100.0 * input_density #(net_size / len(text)) / net_size unique()

    #b = 0.8161848682830884 + h * -0.07234259731194208
    #c = 32.31685851663226 + h * -4.175914850479309

    b = 0.7179041071061821 + h * -0.04398732037836847
    c = 25.220023435454422 + h * -2.0929686503913683

    if add_to_genome:
        set_genome({'TA': h, 'EXP': b, 'S': c})

    return h,b,c




def get_long_text():
    return [' fox eats meat. boy drinks juice. penguin likes ice.']
    #return [' boy drinks juice. penguin likes ice. man drives car.']
    #return [' this is a test to see wether the network can learn longer sequences.']

def split_into_words(sentences):
    result = []
    for s in sentences:
        for w in s.split(' '):
            if w!='':
                result.append(' ' + w)
    return result

#grammar = get_reber_text(10)

#10: MNIST_Patterns_On_Off(center_x=14, class_ids=[10,11,12,13,14], patterns_per_class=1),

    #40: LearningInhibition_test(transmitter='GABA', a='[0.5#a]', b=0.0, c='[50.0#c]', d='[1.5#d]'),#'[0.6410769611853464#a]'

# 0.377 #0.38=np.tanh(0.02 * 20) , threshold=0.38 #np.tanh(get_gene('S',20.0)*get_gene('TA',0.03))