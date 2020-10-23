# https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface

from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer, DataCollator
import json
from config import vocab_size, seq_len, corpus, corpus_path, max_epoch
import os
import torch
import numpy as np

torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)

def tokenize(filename):

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

def load_dataset(path, tokenizer):

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

def train(epoch):

    tokenize(corpus_path)

    tokenizer = load_gpt_tokenizer(corpus)
    # print(tokenizer.tokenize(' Hi there <|endoftext|>'))


    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
    )

    model = GPT2LMHeadModel(config)

    print(f'{model.num_parameters()} parameters')

    dataset, data_collator = load_dataset(path=corpus_path, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f'./{corpus}/trained/checkpoints_{epoch}',
        overwrite_output_dir=True,
        num_train_epochs=epoch,
        per_device_train_batch_size=32,
        save_steps=1575,
        save_total_limit=epoch,
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

    trainer.save_model(f"./{corpus}/trained/checkpoints_{epoch}")


def main():

    for i in range(max_epoch, max_epoch+1):
        print(f'{i} epoch training begins...')
        if not os.path.exists(f'./{corpus}/trained/checkpoints_{i}'):
            os.system(f'mkdir {corpus}/trained/checkpoints_{i}')
        train(epoch=i)

    return




if __name__ == '__main__':

    main()
