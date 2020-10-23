from lang_acq_gpt_train import load_gpt_tokenizer
# from config import corpus, corpus_path
from collections import Counter
import pickle

def prepare_tokenizer(corpus):

    if corpus == 'cbt':
        tokenizer = load_gpt_tokenizer('./cbt')
    elif corpus == 'childes':
        tokenizer = load_gpt_tokenizer('./childes')

    return tokenizer

def prepare_corpus(corpus):

    if corpus == 'childes':
        corpus_path = 'child_directed.txt'
    elif corpus == 'cbt':
        corpus_path = 'cbt_train.txt'

    with open(corpus_path, 'r') as f:
        text = f.read()


    return text

def count(corpus):

    tokenizer = prepare_tokenizer(corpus)
    text = prepare_corpus(corpus)
    encoded = tokenizer.tokenize(text)
    counted = Counter(encoded)
    # print(counted)

    with open(f'counter_{corpus}.pkl', 'wb') as f:
        pickle.dump(counted, f)

    return

def count_without_tokenizer(corpus):
    text = prepare_corpus(corpus)
    text = text.split()
    counted = Counter(text)

    print(counted['goes'])
    print(sum(counted.values()))

    with open(f'counter_{corpus}_split.pkl', 'wb') as f:
        pickle.dump(counted, f)


def load_counter(corpus):

    with open(f'counter_{corpus}.pkl', 'rb') as f:
        counter = pickle.load(f)

    return counter

def main():

    count_without_tokenizer('childes')
    count_without_tokenizer('cbt')
    # count('childes')
    # count('cbt')
    # counter = load_counter('childes')
    #
    # print(counter['went'])
    #
    # print(sum(counter))
    return






if __name__ == '__main__':
    main()

