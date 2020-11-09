import pandas as pd
from config import random_seeds, corpora
from pprint import pprint
from input import Sentences
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

def read_csv(filename):

    data = pd.read_csv(filename, sep='\t', header=None)
    # print(data)

    return data

def get_average(csvs, start_column=0):
    cat = pd.concat(csvs, axis=0)
    cat = cat.groupby(level=0)
    cat = cat.mean()

    output = pd.concat((csvs[0].iloc[:,:start_column], cat), axis=1)
    print(output)


    return output

def run_average(type, start_column):

    for c in corpora:
        csvs = []
        for r in random_seeds:
            csvs.append(read_csv(f'./result/{type} probability {c} {r}.tsv'))
        data = get_average(csvs, start_column=start_column)
        data.to_csv(f'./result/average {type} probability {c}.tsv', sep='\t', header=False, index=False)
        print(f'saved as ./result/average {type} probability {c}.tsv')

    return

def save_average():

    run_average("sentence", start_column=3)
    run_average("question pres", start_column=4)
    run_average("question past", start_column=4)

    return

def get_token_freq_dict(token_freq):

    output = OrderedDict()
    for i, r in token_freq.iterrows():
        output[r[1]] = r[2]
    return output


def draw_sent_prob_submain(keys, data, token_freq, corpus):

    for k in keys:
        logs = [r[1:] for i,r in data.iterrows() if r[0]==k]
        plot_sent_prob(k, logs, token_freq, corpus)

    return

def draw_ques_prob_submain(keys, data, token_freq, corpus, type):
    for k in keys:
        logs = [r[1:] for i,r in data.iterrows() if r[0]==k]
        plot_ques_prob(k, logs, token_freq, corpus, type)

    return


def plot_sent_prob(key, logs, token_freq, corpus):


    fig, ax = plt.subplots()
    epoch_len = len(logs[0][2:])
    sent_plots = []



    for i,s in enumerate(logs):
        # print(s[2:])
        plot, = ax.plot(range(1, epoch_len+1), s[2:], label=s[2])
        sent_plots.append(plot)
        # plot2, = ax2.plot(range(1, epoch_len+1), [token_freq[s[1]]] * epoch_len, linestyle='dashed')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sentence Generation Probability')
    # sent_legend = ax.legend(handles=sent_plots, loc='upper left', fontsize='x-small')
    # plt.gca().add_artist(sent_legend)

    ax2 = ax.twinx()
    token_plots = []
    ax2.set_ylabel('Token Frequency')

    for i,s in enumerate(logs):
        plot2, = ax2.plot(range(1, epoch_len+1), [token_freq[s[1]]] * epoch_len, linestyle='dashed', label=s[1])
        token_plots.append(plot2)


    sent_legend = ax.legend(handles=sent_plots, loc='upper left', fontsize='x-small')
    sent_legend.remove()
    # plt.gca().add_artist(sent_legend)
    token_legend = ax2.legend(handles=token_plots, loc='upper right', fontsize='x-small')
    # token_legend.set_zorder(0)
    plt.gca().add_artist(sent_legend)

    plt.title(f'{corpus} - {key}')

    fig.tight_layout()
    plt.savefig(f'./result/plot/sentence_prob_{corpus}_{key}.png', dpi=300)
    print(f'saved as ./result/plot/sentence_prob_{corpus}_{key}.png')
    plt.close()
    plt.cla()
    plt.clf()

    return

def plot_ques_prob(key, logs, token_freq, corpus, type):


    fig, ax = plt.subplots()
    epoch_len = len(logs[0][3:])
    sent_plots = []



    for i,s in enumerate(logs):
        # print(s[2:])
        plot, = ax.plot(range(1, epoch_len+1), s[3:], label=s[3])
        sent_plots.append(plot)
        # plot2, = ax2.plot(range(1, epoch_len+1), [token_freq[s[1]]] * epoch_len, linestyle='dashed')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sentence Generation Probability')
    # sent_legend = ax.legend(handles=sent_plots, loc='upper left', fontsize='x-small')
    # plt.gca().add_artist(sent_legend)

    ax2 = ax.twinx()
    token_plots = []
    ax2.set_ylabel('Token Frequency')

    for i,s in enumerate(logs):
        plot2, = ax2.plot(range(1, epoch_len+1), [token_freq[s[1]]] * epoch_len, linestyle='dashed', label=s[1])
        token_plots.append(plot2)


    sent_legend = ax.legend(handles=sent_plots, loc='upper left', fontsize='x-small')
    sent_legend.remove()
    # plt.gca().add_artist(sent_legend)
    token_legend = ax2.legend(handles=token_plots, loc='upper right', fontsize='x-small')
    # token_legend.set_zorder(0)
    plt.gca().add_artist(sent_legend)

    plt.title(f'{type.upper()} - {corpus} - {key}')

    fig.tight_layout()
    plt.savefig(f'./result/plot/ques_{type}_prob_{corpus}_{key}.png', dpi=300)
    print(f'saved as ./result/plot/ques_{type}_prob_{corpus}_{key}.png')
    plt.close()
    plt.cla()
    plt.clf()

    return


def draw_sent_prob_main():

    keys = Sentences.keys()
    print(keys)

    for c in corpora:
        token_freq_data = pd.read_csv(f'./result/token frequency {c}.tsv', sep='\t', header=None)
        token_freq_data = get_token_freq_dict(token_freq_data)
        data = pd.read_csv(f'./result/average sentence probability {c}.tsv', sep='\t', header=None)
        draw_sent_prob_submain(keys, data, token_freq_data, c)


    return


def draw_ques_prob_main():

    keys = Sentences.keys()

    for c in corpora:
        token_freq_data = pd.read_csv(f'./result/token frequency {c}.tsv', sep='\t', header=None)
        token_freq_data = get_token_freq_dict(token_freq_data)
        for t in ['pres', 'past']:
            data = pd.read_csv(f'./result/average question {t} probability {c}.tsv', sep='\t', header=None)
            draw_ques_prob_submain(keys, data, token_freq_data, c, type=t)

    return

def create_fake_data(epoch=20):

    overgen_curve = np.concatenate((np.linspace(0.17, 0.3, num=int(epoch/4)), np.linspace(0.3, 0, num=(epoch-int(epoch/4)))))
    # overgen_curve = np.sqrt(overgen_curve)
    correct_past = np.concatenate((np.linspace(0.3, 0.2, num=int(epoch/4)), np.linspace(0.2, 0.4, num=(epoch-int(epoch/4)))))
    double_past = np.concatenate((np.linspace(0.15, 0.13, num=int(epoch/4)), np.linspace(0.13, 0.15, num=(int(epoch/4))),np.linspace(0.15, 0, num=(int(epoch/2)))))

    logs = [
        ['PRES (go)', np.linspace(0.55, 0.51, num=epoch)],
        ['PRES-ed (goed)', overgen_curve],
        ['PRES-s (goes)', np.linspace(0.4, 0.5, num=epoch)],
        ['PAST (went)', correct_past],
        ['PAST-ed (wented)', double_past],
    ]

    # pprint(logs)

    return logs

def draw_hypothe_graph_main(epoch=20):

    logs = create_fake_data(epoch=epoch)

    fig, ax = plt.subplots()
    sent_plots = []

    for i,s in enumerate(logs):
        plot, = ax.plot(np.linspace(0, 1, num=epoch), s[1], label=s[0])
        sent_plots.append(plot)

    ax.set_xlabel('Language Acquisition Process')
    ax.set_ylabel('Sentence Generation Probability')

    plt.xticks([])
    plt.yticks([])

    sent_legend = ax.legend(handles=sent_plots, loc='upper left', fontsize='x-small')


    plt.title(f'Hypothetical Prediction')

    fig.tight_layout()
    plt.savefig(f'./result/plot/sentence_prob_hypothetical.png', dpi=300)
    print(f'saved as ./result/plot/sentence_prob_hypothetical.png')
    plt.close()
    plt.cla()
    plt.clf()


    return


def main():

    save_average()
    # draw_hypothe_graph_main(epoch=200)
    # draw_sent_prob_main()
    draw_ques_prob_main()

    return


if __name__ == '__main__':

    main()