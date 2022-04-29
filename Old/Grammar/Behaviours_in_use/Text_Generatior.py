from PymoNNto import *

def unique(l):
    return list(sorted(set(l)))

def grammar_text_blocks():
    verbs = ['eats ',
                  'drinks ']

    subjects = [' man ',
                     ' woman ',
                     ' girl ',
                     ' boy ',
                     ' child ',
                     ' cat ',
                     ' dog ',
                     ' fox ']


    objects_eat = ['meat.',
                        'bread.',
                        'fish.',
                        'vegetables.']

    objects_drink = ['milk.',
                          'water.',
                          'juice.',
                          'tea.']

    results=[]
    for s in subjects:
        for oe in objects_eat:
            results.append(s + verbs[0] + oe)

        for od in objects_drink:
            results.append(s + verbs[1] + od)

    return results

def grammar_text_blocks_simple():
    verbs = ['eats ',
                  'drinks ']

    subjects = [
                     ' boy ',
                     ' cat ',
                     ' dog ',
                     ' fox ']


    objects_eat = ['meat.',
                        'bread.',
                        'fish.',]

    objects_drink = ['milk.',
                          'water.',
                          'juice.',]

    results=[]
    for s in subjects:
        for oe in objects_eat:
            results.append(s + verbs[0] + oe)

        for od in objects_drink:
            results.append(s + verbs[1] + od)

    return results

class Text_Generator(Behaviour):

    set_variables_on_init = True

    def get_text_blocks(self):
        return self.get_init_attr('text_blocks', [])

    def set_variables(self, neurons):

        self.text_blocks = self.get_text_blocks()
        self.current_block_index = -1
        self.position_in_current_block = -1
        self.alphabet = unique(' '.join(self.text_blocks)) #list not string!!!!
        self.history = ''

        # output
        neurons.current_char = ''
        neurons.current_char_index = -1

        # manual activation
        self.next_char = None

        if self.get_init_attr('set_network_size_to_alphabet_size', False):
            dim = get_squared_dim(len(self.alphabet)) #NeuronDimension
            dim.set_variables(neurons) #set size and x, y, z, width, height, depth

        char_count_vec = self.count_chars_in_blocks()
        self.char_weighting = char_count_vec / np.mean(char_count_vec)


    def new_iteration(self, neurons):

        if self.next_char is not None:# manual activation
            neurons.current_char = self.next_char
            self.next_char = None
        else:
            neurons.current_char = self.get_char(next=True)

        neurons.current_char_index = self.char_to_index(neurons.current_char)

        self.history += neurons.current_char

    def get_char(self, next=False):
        if next:
            self.position_in_current_block += 1

        if self.position_in_current_block >= len(self.text_blocks[self.current_block_index]):
            self.current_block_index=self.get_next_block_index()
            self.position_in_current_block = 0

        return self.text_blocks[self.current_block_index][self.position_in_current_block]

    def index_to_char(self, index):
        return self.alphabet[index]

    def char_to_index(self, char):
        return self.alphabet.index(char)

    def get_next_block_index(self):
        return np.random.randint(len(self.text_blocks))

    def set_next_char(self, char):#manual activation
        self.next_char = char

    def get_words(self):
        return unique([word for word in ' '.join(self.text_blocks).replace('.', ' ').replace(',', ' ').replace('?', ' ').replace('!', ' ').split(' ') if word != ''])

    def count_chars_in_blocks(self):
        result = np.zeros(len(self.alphabet))
        for block in self.text_blocks:
            for c in block:
                result[self.char_to_index(c)] += 1
        return result

    def get_text_score(self, text):
        block_scores = [1 for _ in self.text_blocks]
        for i in range(len(text)):
            for bi, block in enumerate(self.text_blocks):
                block_score = 0
                comp_text = text[i:i+len(block)]
                comp_block = block[0:len(comp_text)]
                for t, b in zip(comp_text, comp_block):
                    if t == b:
                        block_score += 1 * (1/self.char_weighting[self.char_to_index(t)])

                block_scores[bi] += (block_score*block_score)/len(text)
        score = 0
        for bs in block_scores:
            score += np.sqrt(bs)
        return score

