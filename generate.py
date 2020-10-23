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
from config import gen_len, corpus
import csv
from config import max_epoch
from get_token_stats import load_counter



torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)

def prepare_tokenizer_model(epoch):
    if corpus == 'cbt':
        tokenizer = load_gpt_tokenizer('./cbt')
        model = GPT2LMHeadModel.from_pretrained(f'./cbt/trained/checkpoints_20/checkpoint-{1575*epoch}')
    elif corpus == 'childes':
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

    with open(f'generated_{corpus}.tsv', 'w', newline='') as f:
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
        for e in range(1, max_epoch+1):
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

def get_sentences_prob(key, sentences):

    logs = []
    token_logs = []

    sents, tokens = zip(*sentences)
    print(sents)
    print(tokens)

    for s, w in sentences:
        print(f'\nSentence : {s}')
        probs = []
        token_freq = get_token_frequency(w)
        for e in range(1,max_epoch+1):
            prob = sent_prob(s, epoch=e).data.cpu().numpy().item()
            probs.append(prob)
            print(f'epoch{e} probability: {prob}')
            # print(probs)

        # logs = prob_normalize(logs)
        token_logs.append(token_freq)
        logs.append(probs)

    pprint(logs)
    draw_sent_prob(sents, logs, tokens=tokens, token_prob=token_logs, filename=f'{corpus}: {key}', title=f'{corpus}: {key}')

    return logs

def prob_normalize(logs):
    print(logs)

    return logs

def get_token_frequency(token):
    counter = load_counter(corpus+'_split')
    return counter[token] / sum(counter.values())




def sent_prob_main():

    logs = []

    sentences = {'go' :
                     [['I go there.', 'go'],
                     ['I went there.', 'went'],
                     ['I wented there.', 'wented'],
                     ['I goed there.', 'goed'],
                     ['I goes there.','goes']],
                 'eat':
                 [['I eat this.', 'eat'],
                  ['I ate this.', 'ate'],
                  ['I ated this.', 'ated'],
                  ['I eated this.', 'eated'],
                  ['I eats this.', 'eats']],
                 'know':
                     [['I know this.', 'know'],
                      ['I knew this.', 'knew'],
                      ['I knewed this.', 'knewed'],
                      ['I knowed this.', 'knowed'],
                      ['I knows this.', 'knows']],
                 'find':
                     [['I find this.', 'find'],
                      ['I found this.', 'found'],
                      ['I founded this.', 'founded'],
                      ['I finded this.', 'finded'],
                      ['I finds this.', 'finds']],
                 'feel':
                     [['I feel this.', 'feel'],
                      ['I felt this.', 'felt'],
                      ['I felted this.', 'felted'],
                      ['I feeled this.', 'feeled'],
                      ['I feels this.', 'feels']],


                 }

    for key in sentences:
        get_sentences_prob(key, sentences[key])



    return




def main():

    sent_prob_main()
    generate_main()

    return


if __name__=="__main__":

    main()





# 동화책 코퍼스와 비교
# 규칙 불규칙 (실수 많은것, 없는것)
# 들어본적 없는 goed, 인간 패턴 학습, generalization, 단순히 외운다, abstraction 없이, feature 없
# 어른의 실, goed went go goes frequency, sneak, snuck, sneaked, 어른이 자주 실수하는...
# 패턴을 학습하는지 그냥 외우는지,