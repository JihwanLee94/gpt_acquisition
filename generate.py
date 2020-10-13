from transformers import pipeline, GPT2LMHeadModel
from lang_acq_gpt_train import load_gpt_tokenizer
import torch
import numpy as np
from pprint import pprint
from entmax import entmax_bisect
from tqdm import tqdm
from plot import draw_prob_graph, draw_sent_prob
from copy import deepcopy
from sample import entmax, greedy, sample_default, cal_sent_prob
from config import gen_len
import csv



torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)

def prepare_tokenizer_model(epoch):
    tokenizer = load_gpt_tokenizer('./childes')
    model = GPT2LMHeadModel.from_pretrained(f'./trained/checkpoints_{epoch}')

    return tokenizer, model

def generate(prompt, epoch):

    tokenizer, model = prepare_tokenizer_model(epoch)
    input_ids = tokenizer.encode(prompt, return_tensors='pt', pad_token_id=tokenizer.eos_token_id)


    default_output = sample_default(input_ids, tokenizer, model, prompt=prompt, epoch=epoch, max_length=gen_len)
    greedy_output = greedy(input_ids, tokenizer, model, prompt=prompt, epoch=epoch, max_length=gen_len)
    entmax_output = entmax(input_ids, tokenizer, model, epoch=epoch, prompt=prompt, max_length=gen_len)


    print(f'default\t----------------\n {default_output}')
    print(f'greedy\t----------------\n {greedy_output}')
    print(f'entmax\t----------------\n {entmax_output}')

    return default_output, greedy_output, entmax_output



def to_tsv(logs):

    with open('generated.tsv', 'w', newline='') as f:
        tsv_output = csv.writer(f, delimiter='\t')
        tsv_output.writerows(logs)

    print('saved as tsv')

    return

def generate_main():

    prompts = ['Where did he go yesterday? ',
               'There is a wug. There are two ']

    decoding = ['default',
                'greedy',
                'entmax']

    txt_logs = [['prompt', 'epoch', 'decoding', 'generated']]

    for p in prompts:
        print(f'\nPrompt : {p}')
        for e in range(1, 11):
            print(f'\nepoch: {e}')
            generated = generate(p, epoch=e)
            for i,d in enumerate(decoding):
                txt_logs.append([p, e, d, generated[i]])



    # pprint(txt_logs)
    to_tsv(txt_logs)

    return

def sent_prob(sent, epoch):

    tokenizer, model = prepare_tokenizer_model(epoch)
    input_ids = tokenizer.encode(sent, return_tensors='pt', pad_token_id=tokenizer.eos_token_id)

    prob = cal_sent_prob(input_ids, tokenizer=tokenizer, model=model, sentence=sent, epoch=epoch)

    return prob

def sent_prob_main():

    logs = []

    sentences = ['He goes there.',
                 'He went there.',
                 'He goed there.',
                 'He go there.',]

    for s in sentences:
        print(f'\nSentence : {s}')
        probs = []
        for e in range(1,11):
            prob = sent_prob(s, epoch=e).data.cpu().numpy().item()
            probs.append(prob)
            print(f'epoch{e} probability: {prob}')
            # print(probs)

        logs.append(probs)

    pprint(logs)
    draw_sent_prob(sentences, logs, filename='go', title='go')


    return




def main():

    sent_prob_main()
    generate_main()

    return


if __name__=="__main__":

    main()