# vocab_size = 10000 # five year old children
# seq_len = 128
# gen_len = 50
# max_epoch = 20

random_seeds = [999, 15, 777]
corpora = ['cbt', 'childes']

# random_seeds = [777]
# corpora = ['childes']

class Config:
    def __init__(self, random_seed, corpus, vocab_size=10000, seq_len=128, gen_len=50, max_epoch=20):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.gen_len = gen_len
        self.max_epoch = max_epoch
        self.random_seed = random_seed
        self.corpus = corpus
        if corpus == 'childes':
            self.corpus_path = 'child_directed.txt'
            self.step = 3030
        elif corpus == 'cbt':
            self.corpus_path = 'cbt_train.txt'
            self.step = 1575

        print('corpus: ', self.corpus)
        print('random seed: ', self.random_seed)


# config = config()

# corpus = 'cbt'
# corpus = 'childes'

# config = config
#
# print('corpus: ', config.corpus)
# print('random seed: ', config.random_seed)


# if corpus == 'childes':
#     corpus_path = 'child_directed.txt'
# elif corpus == 'cbt':
#     corpus_path = 'cbt_train.txt'



