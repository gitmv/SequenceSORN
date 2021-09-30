def unique(l):
    return list(sorted(set(l)))

def mark_with_grammar(str, sentences, html=True):

    all_sentences = ''.join(sentences)
    words = all_sentences.split(' ')
    words.pop(0)

    #print(sentences)
    #print(words)

    output_sentences = [s+'.' for s in str.split('.')]

    #print(output_sentences)

    new_sentences = []

    for sentence in output_sentences:

        words_in_sentence = sentence.split(' ')
        words_in_sentence.pop(0)

        print(words_in_sentence)

        wrong = False
        for word in words_in_sentence:
            if word not in words:
                wrong = True

        if sentence not in sentences and ' ' + sentence not in sentences:
            if not wrong and len(words_in_sentence) >= 3:
                if sentence not in new_sentences:
                    new_sentences.append(sentence)

    mark_dict = {}

    mark_dict["background-color:#00FF00"] = sentences  # right
    mark_dict["background-color:#FFFF00"] = new_sentences  # new
    mark_dict["color:#0000FF"] = [' ' + w for w in words]  # right words

    #print(mark_dict)

    return mark_text(str, mark_dict, html)


def mark_text(str, styles, html=True):  # styles={"background-color:#FF0000":[text1,text2,...], }
    result = str
    tags = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
    i = 0
    for style, mark_str_list in styles.items():

        for mark_str in mark_str_list:
            result = result.replace(mark_str,
                                    '<' + tags[i] + ' style="' + style + '";>' + mark_str + '</' + tags[i] + '>')

        i += 1

    if html:
        return """<html>
        <head></head>
        <body>""" + result + """</body>
        </html>"""
    else:
        return result


t='s juin likes ice. fox eaty drinks juin uin uin likenguin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin uin likes ice. boy enguin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin likenguin likes ice. fox eats meat. boy drinks juin likes y iy fox e. boy y y enguin likes ice. fox eats meat. boy drinks juin likes ice. boy drinks juin like. fox enguin likes ice. fox eats meat. boy drinks juin likes ice. box enguin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin uin likes ice. fox eats meat. boy drinks juin uin likes ice. boy drinks juin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy dringuin likes ice. fox eats meat. boy drinks juin uin likes ice. fox eats meat. boy drinks juin uin likenguin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy enguin likes iy fox eat. penguin likes ice. fox eats meat. boy drinks juin likes enguin likes ice. fox eats meat. boy drinks juin like. fox e. y x enguin likes ice. fox eat. fox enguin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin uin like. fox enguin likes ice. fox eats meat. boy drinks juin likes ice. fox eaty enguin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin uin uinguin likes ice. fox eats meat. boy drinks juine. boy drinks juin like. foy dringuin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin likes ice. boy dringuin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin likenguin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juice. fox eats meat. boy drinks juin uin uin uinguin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin like. fox eat. penguin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juice. fox eats meat. boy drinks juin likes juin likenguin likes ice. fox eats meat. boy drinks juin uin uinguin likes ice. fox eats meat. boy drinks juin lice. fox eat. boy enguin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juine. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juine. fox eats meat. boy drinks juin likenguin likes ice. fox eats meat. boy drinks juine. fox enguin likes ice. fox eats meat. boy drinks juice. fox eats meat. boy drinks juine. fox eats ice. fox eats meat. boy drinks juin likes iy drinks juinguin likes ice. fox eats meat. boy drinks juin likes ice. fox enguin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin uice. fox eaty y ice. fox eats ice. fox eats meat. boy drinks juinguin likes ice. fox eats meat. boy drinks juin likenguin likes ice. fox eats meat. boy drinks juin uin uinguin likes ice. fox eats meat. boy drinks juice. fox eats meat. boy drinks juin likes iy foy enguin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin like. fox enguin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin uice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin likenguin likes ice. fox eats meat. boy drinks juice. fox eat. foy drinkenguin likes ice. fox eats meat. boy dringuin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juice. fox eats iy enguin likes ice. fox eats meat. boy drinks juice. fox eats meat. boy drinks juin likes ice. fox eat. ice. fox enguin likes ice. fox eats meat. boy drinks juice. fox eaty dringuin likes ice. fox eat. juinguin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juin likes ice. fox eats meat. boy drinks juinguin likes ice. fox eats meat. boy drinks juice. fox eat. boy drinks juice. fox eats meat. box enguin likes ice. fox eats meat. boy drinks juin uinguin likes ice. fox eats meat. boy drinks juinguin likes ice. fox eats meat. boy drinks juine. box eaty y . boy y y enguin likes ice. fox eats meat. boy drinks juice. fox eats meat. boy drinks juin uin likes ice. fox eats ice. fox enguin likes ice. fox eats meat. boy drinks juice. fox eats meat. boy drinks juin likes ice. fox eat. boy dringuin likes ice. fox eat. likes ice. fox enguin likes ice. fox eats ice. fox enguin likes ice. fox eats meat. boy drinks juice. fox eats'

print(mark_with_grammar(t, [' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.'],True))