from transformers import GPT2LMHeadModel
from lang_acq_gpt_train import load_gpt_tokenizer, set_random_seed
import torch
import numpy as np
from pprint import pprint
from entmax import entmax_bisect
from tqdm import tqdm
from plot import draw_prob_graph, draw_sent_prob
from copy import deepcopy
from sample import entmax, greedy, sample_default, cal_sent_prob, cal_ques_prob
# from config import gen_len, corpus
import csv
from config import Config, random_seeds, corpora
from get_token_stats import load_counter
from input import Sentences, Questions_pres, Questions_past



# torch.manual_seed(777)
# torch.cuda.manual_seed_all(777)
# np.random.seed(777)

def prepare_tokenizer_model(epoch, corpus, random_seed, step):

    print(f'preparing {corpus} corpus tokenizer and model, seed {random_seed}')
    tokenizer = load_gpt_tokenizer(f'./{corpus}')
    model = GPT2LMHeadModel.from_pretrained(f'../../../home_kahlo/jihwan.lee/lang_acquisition/trained/{corpus}/{random_seed}/checkpoints/checkpoint-{step*epoch}')
    # if corpus == 'cbt':
    #     tokenizer = load_gpt_tokenizer('./cbt')
    #     # model = GPT2LMHeadModel.from_pretrained(f'./cbt/trained/checkpoints_20/checkpoint-{1575*epoch}')
    #     model = GPT2LMHeadModel.from_pretrained(f'../../cbt/trained/checkpoints_20/checkpoint-{1575*epoch}')
    #
    # elif corpus == 'childes':
    #     tokenizer = load_gpt_tokenizer('./childes')
    #     model = GPT2LMHeadModel.from_pretrained(f'./trained/checkpoints_{epoch}')

    return tokenizer, model

def generate(prompt, epoch, gen_len):

    tokenizer, model = prepare_tokenizer_model(epoch)
    input_ids = tokenizer.encode(prompt, return_tensors='pt', pad_token_id=tokenizer.eos_token_id)


    default_output = sample_default(input_ids, tokenizer, model, prompt=prompt, epoch=epoch, max_length=gen_len)
    greedy_output = greedy(input_ids, tokenizer, model, prompt=prompt, epoch=epoch, max_length=gen_len)
    entmax_output = entmax(input_ids, tokenizer, model, epoch=epoch, prompt=prompt, max_length=gen_len)


    print(f'default\t----------------\n {default_output}')
    print(f'greedy\t----------------\n {greedy_output}')
    print(f'entmax\t----------------\n {entmax_output}')

    return default_output, greedy_output, entmax_output



def to_tsv(logs, filename):

    pprint(logs)

    with open(f'./result/{filename}.tsv', 'w', newline='') as f:
        tsv_output = csv.writer(f, delimiter='\t')
        tsv_output.writerows(logs)

    print(f'{filename} saved as tsv')

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
    to_tsv(txt_logs, f'generated_{corpus}')

    return

def sent_prob(sent, epoch, corpus, random_seed, step):

    tokenizer, model = prepare_tokenizer_model(epoch, corpus=corpus, random_seed=random_seed, step=step)
    input_ids = tokenizer.encode(sent, return_tensors='pt', pad_token_id=tokenizer.eos_token_id)

    prob = cal_sent_prob(input_ids, tokenizer=tokenizer, model=model, sentence=sent, epoch=epoch)

    return prob

def ques_prob(prompt, sent, epoch, corpus, random_seed, step):
    tokenizer, model = prepare_tokenizer_model(epoch, corpus=corpus, random_seed=random_seed, step=step)
    print(prompt, sent)
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt', pad_token_id=tokenizer.eos_token_id)
    sent_ids = tokenizer.encode(sent, return_tensors='pt', pad_token_id=tokenizer.eos_token_id)

    prob = cal_ques_prob(prompt_ids, sent_ids, model)

    return prob


def get_sentences_prob(config, key, sentences):

    logs = []
    token_logs = []
    to_csv = []

    sents, tokens = zip(*sentences)
    print(sents)
    print(tokens)

    for s, w in sentences:
        print(f'\nSentence : {s}')
        probs = [key, w, s, ]
        # token_freq = [key, w, ]
        token_freq=get_token_frequency(w, config.corpus)
        for e in range(1,config.max_epoch+1):
            prob = sent_prob(s, epoch=e, corpus=config.corpus, random_seed=config.random_seed, step=config.step).data.cpu().numpy().item()
            probs.append(prob)
            print(f'epoch{e} probability: {prob}')
            # print(probs)

        # logs = prob_normalize(logs)
        token_logs.append(token_freq)
        logs.append(probs[3:])
        to_csv.append(probs)

    # pprint(logs)
    pprint(to_csv)
    # draw_sent_prob(sents, logs, tokens=tokens, token_prob=token_logs, filename=f'{corpus}: {key}', title=f'{corpus}: {key}')

    return to_csv

def get_ques_prob(config, key, sentences):

    logs = []

    for prompt, s, w in sentences:
        probs = [key, w, prompt, s, ]
        for e in range(1, config.max_epoch+1):
            prob = ques_prob(prompt=prompt, sent=s, epoch=e, corpus=config.corpus, random_seed=config.random_seed, step=config.step).data.cpu().numpy().item()
            probs.append(prob)
            print(f'epoch{e} probability: {prob}')

        logs.append(probs)

    pprint(logs)



    return logs

def prob_normalize(logs):
    print(logs)

    return logs

def get_token_frequency(token, corpus):
    counter = load_counter(corpus+'_split')
    return 1000 * counter[token] / sum(counter.values())

def save_token_freq(corpus, sents, key):

    token_log = []
    for s, w in sents:
        token_freq = [key, w, get_token_frequency(w, corpus)]
        token_log.append(token_freq)

    # print('token log')
    # pprint(token_log)

    return token_log

# def log_to_csv(log, filename):
#     with open(filename, 'w') as f:
#         wr = csv.writer(f, quoting=csv.QUOTE_ALL)
#         wr.writerow(log)
#
#     return


def sent_prob_main(config):

    set_random_seed(config.random_seed)

    logs = []
    token_freq_log = []

    sentences = deepcopy(Sentences)

    # sentences = {'go' :
    #                  [['I go there.', 'go'],
    #                  ['I goed there.', 'goed'],
    #                  ['I goes there.', 'goes'],
    #                  ['I went there.', 'went'],
    #                  ['I wented there.','wented']],
    #              'eat':
    #              [['I eat this.', 'eat'],
    #               ['I eated this.', 'eated'],
    #               ['I eats this.', 'eats'],
    #               ['I ate this.', 'ate'],
    #               ['I ated this.', 'ated']],
    #              'know':
    #                  [['I know this.', 'know'],
    #                   ['I knowed this.', 'knowed'],
    #                   ['I knows this.', 'knows'],
    #                   ['I knew this.', 'knew'],
    #                   ['I knewed this.', 'knewed']],
    #              'write':
    #                  [['I write this.', 'write'],
    #                   ['I writed this.', 'writed'],
    #                   ['I writes this.', 'writes'],
    #                   ['I wrote this.', 'wrote'],
    #                   ['I wroted this.', 'wroted']],
    #              'feel':
    #                  [['I feel this.', 'feel'],
    #                   ['I feeled this.', 'feeled'],
    #                   ['I feels this.', 'feels'],
    #                    ['I felt this.', 'felt'],
    #                 ['I felted this.', 'felted']],
    #
    #             ######## regular ########
    #             'like':
    #                  [['I like this.', 'like'],
    #                   ['I liked this.', 'liked'],
    #                   ['I likes this.', 'likes']],
    #
    #              'hate':
    #                  [['I hate this.', 'hate'],
    #                   ['I hated this.', 'hated'],
    #                   ['I hates this.', 'hates']],
    #
    #              'want':
    #                  [['I want this.', 'want'],
    #                   ['I wanted this.', 'wanted'],
    #                   ['I wants this.', 'wants']],
    #
    #
    #
    #              }

    if config.random_seed == 999: ## run only once
        for key in sentences:
            token_freq_log.extend(save_token_freq(corpus=config.corpus, key=key, sents=sentences[key]))
        to_tsv(token_freq_log, f'token frequency {config.corpus}')

    for key in sentences:
        logs.extend(get_sentences_prob(config, key, sentences[key]))

    to_tsv(logs, f'sentence probability {config.corpus} {config.random_seed}')



    return

def ques_prob_main(config):
    set_random_seed(config.random_seed)
    logs_pres = []
    logs_past = []

    questions_pres = deepcopy(Questions_pres)
    questions_past = deepcopy(Questions_past)


    for key in questions_pres:
        logs_pres.extend(get_ques_prob(config, key, questions_pres[key]))
        logs_past.extend(get_ques_prob(config, key, questions_past[key]))

    to_tsv(logs_past, f'question past probability {config.corpus} {config.random_seed}')
    to_tsv(logs_pres, f'question pres probability {config.corpus} {config.random_seed}')


    return




def main():

    for r in random_seeds:
        for c in corpora:
            config = Config(random_seed=r, corpus=c)
            ques_prob_main(config)
            # sent_prob_main(config)
            # generate_main(config)

    return


if __name__=="__main__":

    main()





# 동화책 코퍼스와 비교
# 규칙 불규칙 (실수 많은것, 없는것)
# 들어본적 없는 goed, 인간 패턴 학습, generalization, 단순히 외운다, abstraction 없이, feature 없
# 어른의 실, goed went go goes frequency, sneak, snuck, sneaked, 어른이 자주 실수하는...
# 패턴을 학습하는지 그냥 외우는지,