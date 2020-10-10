from transformers import pipeline, GPT2LMHeadModel
from lang_acq_gpt_train import load_gpt_tokenizer
import torch
import numpy as np
from pprint import pprint
from entmax import entmax_bisect
from tqdm import tqdm
from plot import draw_prob_graph
from copy import deepcopy
from sample import entmax, greedy
from config import gen_len
import csv



torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)

def generate(prompt, epoch):

    # generated = pipeline(task='text-generation', model='./checkpoints', tokenizer='./childes', config={'max_length':50})

    tokenizer = load_gpt_tokenizer('./childes')
    model = GPT2LMHeadModel.from_pretrained(f'./trained/checkpoints_{epoch}')

    input_ids = tokenizer.encode(prompt, return_tensors='pt', pad_token_id=tokenizer.eos_token_id)


    # greedy_output = model.generate(input_ids, max_length=50)
    # greedy_output = tokenizer.decode(greedy_output[0], skip_special_tokens=False)
    greedy_output = greedy(input_ids, tokenizer, model, prompt=prompt, epoch=epoch, max_length=gen_len)
    entmax_output = entmax(input_ids, tokenizer, model, epoch=epoch, prompt=prompt, max_length=gen_len)
    # print(f'entmax: {entmax_output}')
    # print('greedy:\n', greedy)
    # print('entmax:\n', entmax)

    return greedy_output, entmax_output



def to_tsv(logs):

    with open('generated.tsv', 'w', newline='') as f:
        tsv_output = csv.writer(f, delimiter='\t')
        tsv_output.writerows(logs)

    print('saved as tsv')

    return

def main():
    prompts = ['Where did he go yesterday? ',
               'There is a wug. There are two ']

    decoding = ['greedy',
                'entmax']

    txt_logs = [['prompt', 'epoch', 'decoding', 'generated']]

    for p in prompts:
        print(f'\nPrompt : {p}')
        for e in range(1, 11):
            print(f'\nepoch: {e}')
            generated = generate(p, epoch=e)
            for i,d in enumerate(decoding):
                txt_logs.append([p, e, d, generated[i]])



    pprint(txt_logs)
    to_tsv(txt_logs)



            # to_txt(p, e)
            # generate('Where did he go yesterday? ', epoch=e)
            # generated = generate('There is a wug. There are two', epoch=e)
    return


if __name__=="__main__":

    main()