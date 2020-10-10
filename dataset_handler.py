import pickle
from pprint import pprint
from copy import deepcopy
import re
import json

def read_file(filename):

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data

def remove_child(data):
    # removed = []
    for d in data:
        removed = []
        for l in d['dialogue']:
            # print(l[0][1:-1], d['target_child'])

            if l[0][1:-1] not in d['target_child']:
                removed.append(l)

        d['dialogue'] = deepcopy(removed)

    return

def clean(text):
    #
    # matched = re.compile('[())]').findall(text)
    # if matched:
    #     print(matched)
    #     print(text)

    # text = re.sub('\x15-?\d+_\d+\x15', '', text) #useless
    text = re.sub('\x15.*\x15', '', text)
    text = re.sub('\[.*\]', '', text) # erase all in []
    # text = re.sub('\[=! \S+\]', '', text) #useless
    text = re.sub('&=\S+ ', ' ', text)
    text = re.sub('@\S+ ', ' ', text)
    text = re.sub('\+[\.\?\/!,"\^\+<]+', ' ', text)
    text = re.sub('\(\.+\)', ' ', text)
    text = re.sub('[\(\)\^:ˌˈ↑↓;<>‡„]', '', text)
    # text = re.sub('<\S+> ', ' ', text )
    text = re.sub('&-', '', text)

    # text = re.sub('\[[!?]+\]', ' ', text)
    # text = re.sub('\[# \d+\.\d+\]', ' ', text)



    # text = re.sub("[()\[\]<>]", "", text)

    # if matched:
    #     print(text)
    #     print()
    return text

def preprocess(data):
    for d in data:
        for l in d['dialogue']:
            # print(l)
            l[1] = clean(l[1]) + '\n'

    return

def to_txt(data):

    text = ''

    for d in data:
        tmp = ' '.join([s[1] for s in d['dialogue']])
        text += tmp
        text += '<|endoftext|>\n'


    # print(text[:-30])

    return text



if __name__ == '__main__' :

    filename = './dataset/talkbank/childes.pkl'
    data = read_file(filename)
    # pprint(data[:10])

    print('removing CHI')
    remove_child(data)
    print('preprocessing')
    preprocess(data)
    with open('preprocessed.json', 'w') as f:
        json.dump(data, f)
    print('saved json')

    text = to_txt(data)
    with open('child_directed.txt', 'w') as f:
        f.write(text)
    print('saved txt')


    # pprint()
    # pprint(data[1234:1244])

    # for d in data:
    #     if len(d['target_child']) < 1:
    #         print(d['path'])
    #         print(d['participants'])
    #         print(d['target_child'])