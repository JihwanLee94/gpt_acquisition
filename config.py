

vocab_size = 10000 # five year old children
seq_len = 128
gen_len = 50
max_epoch = 20

# corpus = 'cbt'
corpus = 'childes'


if corpus == 'childes':
    corpus_path = 'child_directed.txt'
elif corpus == 'cbt':
    corpus_path = 'cbt_train.txt'



