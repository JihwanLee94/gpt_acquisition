# https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface

from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer, DataCollator
import json
# from config import vocab_size, seq_len, corpus, corpus_path, max_epoch
from config import Config, random_seeds, corpora
import os
import torch
import numpy as np

# torch.manual_seed(777)
# torch.cuda.manual_seed_all(777)
# np.random.seed(777)

def tokenize(filename, vocab_size):

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=filename, vocab_size=vocab_size, min_frequency=2, special_tokens=['<|endoftext|>'])
        # '<bos>', '<eos>', '<unk>', '<pad>', '<mask>'])
    tokenizer.save(corpus)

    return tokenizer

def load_gpt_tokenizer(path):

    tokenizer = GPT2Tokenizer(vocab_file=f'./{path}/vocab.json',
                              merges_file=f'./{path}/merges.txt',)
                              # unk_token='<unk>',
                              # bos_token='<bos>',
                              # eos_token='<eos>')
    return tokenizer

def load_dataset(path, tokenizer, seq_len):

    print('loading dataset')
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=path,
        block_size=seq_len
    )

    print('dataset loaded')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    print('data collator ready')

    return dataset, data_collator

def train(config):

    # tokenize(filename=config.corpus_path, vocab_size=config.vocab_size)

    tokenizer = load_gpt_tokenizer(config.corpus)
    # print(tokenizer.tokenize(' Hi there <|endoftext|>'))


    gpt_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.seq_len,
        n_ctx=config.seq_len,
    )

    model = GPT2LMHeadModel(gpt_config)

    print(f'{model.num_parameters()} parameters')

    dataset, data_collator = load_dataset(path=config.corpus_path, tokenizer=tokenizer, seq_len=config.seq_len)

    training_args = TrainingArguments(
        output_dir=f'../../../home_kahlo/jihwan.lee/lang_acquisition/trained/{config.corpus}/{config.random_seed}/checkpoints',
        overwrite_output_dir=True,
        num_train_epochs=config.max_epoch,
        per_device_train_batch_size=32,
        save_steps=config.step,
        seed=config.random_seed,
        # max_steps=
        # save_total_limit=epoch,
    )
    print('training args set')

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    print('start training')

    trainer.train()

    trainer.save_model(f"../../../home_kahlo/jihwan.lee/lang_acquisition/trained/{config.corpus}/{config.random_seed}")

    return

# def sub_main(config):
#
#     # for i in range(max_epoch, max_epoch+1):
#     #     print(f'{i} epoch training begins...')
#     #     if not os.path.exists(f'./{corpus}/trained/checkpoints_{i}'):
#     #         os.system(f'mkdir {corpus}/trained/checkpoints_{i}')
#     # train(config)
#
#     return

def set_random_seed(r):
    torch.manual_seed(r)
    torch.cuda.manual_seed_all(r)
    np.random.seed(r)
    return

def main():

    for r in random_seeds:
        for c in corpora:

            set_random_seed(r)
            config = Config(random_seed=r, corpus=c)
            train(config)


if __name__ == '__main__':

    main()
