import matplotlib.pyplot as plt
from config import corpus

def draw_prob_graph(log, text=None, filename=None, title=None):

    tokens = [t[0] for t in log]
    probs = [p[1] for p in log]

    if filename is None:
        filename = ''.join(tokens[:10])

    fig, ax = plt.subplots()
    fig.set_size_inches(len(tokens)/7, 8)

    if title is not None:
        plt.title(title)
        filename = title + '_' + filename

    ax.plot(range(len(tokens)), probs)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation='vertical', fontsize=9)
    ax.set_ylim(-0.02,1.02)
    if text is not None:
        ax.set_xlabel(text)
    ax.autoscale(False)
    # ax.set_ybound(lower=0, upper=1)
    # plt.ylim([0,1])
    # .tick_params(axis='x', labelsize=8, labelrotation=90)
    # plt.xtics(tokens)
    # plt.show()
    plt.savefig('./plot/'+corpus+'_'+filename+'.png', dpi=300)
    plt.close()

    return


def draw_sent_prob(sents, logs, tokens=None, token_prob=None, filename=None, title=None):

    fig, ax = plt.subplots()
    epoch_len = len(logs[0])
    sent_plots = []
    for i,s in enumerate(sents):
        plot, = ax.plot(range(1,epoch_len+1), logs[i], label=s)
        sent_plots.append(plot)
    print(sent_plots)

    # sent_legend = ax.legend(handles=sent_plots, loc='upper left')
    # plt.gca().add_artist(sent_legend)
    ax.set_xlabel('epoch')
    ax.set_ylabel('sentence probability')
    sent_legend = ax.legend(handles=sent_plots, loc='upper left', fontsize='x-small')
    plt.gca().add_artist(sent_legend)


    if token_prob is not None:
        ax2 = ax.twinx()
        token_plots = []
        ax2.set_ylabel('token frequency')
        for i,w in enumerate(tokens):
            plot, = ax2.plot(range(1, epoch_len+1), [token_prob[i]] * epoch_len, linestyle='dashed', label=w)
            token_plots.append(plot)


    token_legend = ax2.legend(handles=token_plots, loc='upper right', fontsize='x-small')
    plt.gca().add_artist(token_legend)


    fig.tight_layout()
    plt.title(f'{corpus} - {tokens[0]}')
    plt.savefig(f'./plot/sentence_prob_{filename}.png', dpi=300)
    plt.close()
    plt.cla()
    plt.clf()
    return