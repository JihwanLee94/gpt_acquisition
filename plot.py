import matplotlib.pyplot as plt


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
    plt.savefig('./plot/'+filename+'.png', dpi=300)
    plt.close()

    return


def draw_sent_prob(sents, logs, filename=None, title=None):

    fig, ax = plt.subplots()
    epoch_len = len(logs[0])
    for i,s in enumerate(sents):
        ax.plot(range(1,epoch_len+1), logs[i], label=s)

    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('sentence probability')
    plt.savefig(f'./plot/sentence_prob_{filename}.png', dpi=300)
    plt.close()
    return