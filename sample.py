import tensorflow as tf
import tensorflow_probability as tfp
import torch
from entmax import entmax_bisect
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
from plot import draw_prob_graph

torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
np.random.seed(777)


def entmax(input_ids, tokenizer, model, prompt, epoch=None, alpha=1.5, max_length=50):

    new_input_ids = deepcopy(input_ids)
    alpha = torch.tensor(alpha, requires_grad=True)

    log = []

    # print(input_ids)

    for _ in range(max_length):

        prediction_scores = model(new_input_ids)[0][0][-1]
        prediction_prob = entmax_bisect(prediction_scores, alpha)
        candidates = torch.nonzero(prediction_prob)
        next_token_id = candidates[torch.randint(candidates.size()[0], (1,))]

        # print(tokenizer.decode(new_input_ids[0], skip_special_tokens=False))
        # print(tokenizer.decode(next_token_id[0], skip_special_tokens=False),':\t', prediction_prob[next_token_id].data[0][0])

        new_input_ids = torch.cat((new_input_ids, next_token_id), dim=1)

        log.append((tokenizer.decode(next_token_id[0], skip_special_tokens=False), prediction_prob[next_token_id].item()))


    # pprint(log)
    output_sent = tokenizer.decode(new_input_ids[0], skip_special_tokens=False)
    # if epoch is not None:
    #     prompt = f'epoch{epoch}_{prompt}'
    draw_prob_graph(log, text=output_sent, filename=prompt, title=f'GPT entmax epoch{epoch}')

    return output_sent


def greedy(input_ids, tokenizer, model, prompt, epoch=None, max_length=50):
    # print(input_ids)
    log = []
    new_input_ids = deepcopy(input_ids)
    for _ in range(max_length):

        prediction_scores = model(new_input_ids)
        # print(tokenizer.decode(new_input_ids[0], skip_special_tokens=False))
        top_k, next_id = get_top_k_prob(new_input_ids, prediction_scores, tokenizer, k=5)
        # print("topk:", top_k[0][0], tokenizer.encode(top_k[0][0], return_tensors='pt'))
        new_input_ids = torch.cat((new_input_ids, next_id.reshape((1,1))), 1)
        # print("new input ids: ", new_input_ids)
        # get_top_k_sim(top_k)

        log.append(top_k[0])

    # print(tokenizer.decode(new_input_ids[0], skip_special_tokens=False))
    output_sent = tokenizer.decode(new_input_ids[0], skip_special_tokens=False)
    draw_prob_graph(log, text=output_sent, filename=prompt, title=f'GPT greedy epoch{epoch}')

    return output_sent

def sample_default(input_ids, tokenizer, model, prompt, epoch=None, max_length=50):
    # print(input_ids)
    log = []
    new_input_ids = deepcopy(input_ids)
    for _ in range(max_length):

        prediction_scores = model(new_input_ids)
        prediction_scores_softmax = torch.nn.Softmax()(prediction_scores[0][0][-1])/1.0001
        prediction_scores_softmax = prediction_scores_softmax.detach().numpy()
        next_token_id = torch.from_numpy(np.array(np.argmax(np.random.multinomial(1, prediction_scores_softmax, 1)), dtype=np.int64))
        next_token_id = torch.reshape(next_token_id, [1,1])
        new_input_ids = torch.cat((new_input_ids, next_token_id), 1)

        next_token = tokenizer.decode([next_token_id], skip_special_tokens=False)
        next_token_prob = prediction_scores_softmax[next_token_id]

        log.append((next_token, next_token_prob))

    output_sent = tokenizer.decode(new_input_ids[0], skip_special_tokens=False)
    draw_prob_graph(log, text=output_sent, filename=prompt, title=f'GPT default epoch{epoch}')

    return output_sent

def cal_sent_prob(input_ids, tokenizer, model, sentence, epoch):

    prob = 1.0
    for i,idx in enumerate(input_ids[0]):
        temp_input_ids = deepcopy(input_ids[0][:i+1].reshape((1,i+1)))
        # print(i, temp_input_ids)
        prediction_scores = model(temp_input_ids)
        prediction_scores = torch.nn.Softmax()(prediction_scores[0][0][-1])
        if i < len(input_ids[0])-1:
            next_token_prob = prediction_scores[input_ids[0][i+1]]
            prob *= next_token_prob

    return prob ** (1/(len(input_ids[0])-1))

def cal_ques_prob(prompt_ids, sent_ids, model):

    # prob = 1.0

    temp_input_ids = deepcopy(prompt_ids)
    prediction_scores = model(temp_input_ids)
    prediction_scores = torch.nn.Softmax()(prediction_scores[0][0][-1])
    next_token_prob = prediction_scores[sent_ids[0][0]]
    prob = next_token_prob

    for i, idx in enumerate(sent_ids[0]):

        temp_input_ids = torch.cat((prompt_ids, sent_ids[0][:i+1].reshape((1,i+1))),1)
        if i < len(sent_ids[0]) - 1 :
            prediction_scores = model(temp_input_ids)
            prediction_scores = torch.nn.Softmax()(prediction_scores[0][0][-1])
            next_token_prob = prediction_scores[sent_ids[0][i+1]]
            prob *= next_token_prob

    return prob ** (1/(len(sent_ids[0])-1))




def top_s(input_ids, tokenizer, model, max_length=50):

    new_input_ids = deepcopy(input_ids)
    sim_input_ids = deepcopy(input_ids)

    for _ in range(max_length):

        prediction_scores = model(new_input_ids)
        print("Greedy: ",tokenizer.decode(new_input_ids[0], skip_special_tokens=False))
        top_k = get_top_k_prob(new_input_ids, prediction_scores, tokenizer, k=5)
        # get_top_k_sim(top_k)
        new_input_ids = tf.concat([new_input_ids, tf.reshape(tf.math.argmax(prediction_scores[0], axis=2, output_type=tf.dtypes.int32)[0][-1],[1,1])], axis=1)

    print(tokenizer.decode(new_input_ids[0], skip_special_tokens=False))
    greedy_output = model.generate(input_ids, max_length=max_length)
    output_sent = tokenizer.decode(greedy_output[0], skip_special_tokens=False)
    return output_sent

def get_top_k_prob(input_ids, prediction_scores, tokenizer, k=50, temperature=1):
    # print('\nCurrent Sentence:\t',tokenizer.decode(input_ids[0], skip_special_tokens=False))
    prediction_scores = prediction_scores[0][0][-1]
    prediction_scores = torch.nn.Softmax()(prediction_scores/temperature)
    top_k = torch.topk(prediction_scores, k=k, sorted=True)
    top_1_id = top_k[1][0]
    top_k_prob = top_k[0].detach().numpy()
    top_k_decoded = [tokenizer.decode([top_k[1][j]], skip_special_tokens=False) for j in range(k)]
    top_k = list(zip(top_k_decoded, top_k_prob))
    # print('Candidates:')
    # pprint(top_k)
    # print(prediction_scores)
    return top_k, top_1_id

def get_top_k_sim(top_k, k=5):

    word_embeddings = model.transformer.wte.weight[:].numpy()
    # print(word_embeddings)

    top = top_k[0][0]
    top_id = tokenizer.encode([top], return_tensors='tf')
    top_embedding = model.transformer.wte.weight[top_id[0][0],:]
    cos_sim = cal_cos_sim(top_embedding, word_embeddings)
    top_s = tf.math.top_k(cos_sim, k=k, sorted=True)
    top_s_sim = top_s[0].numpy()
    top_s_decoded = [tokenizer.decode([top_s[1][j]], skip_special_tokens=False) for j in range(k)]
    top_s = list(zip(top_s_decoded, top_s_sim))
    print(f'Similar tokens to {[top]}:')
    pprint(top_s)
    # top_embedding = tf.reshape(top_embedding, shape=[1,768])
    # top_embedding = tf.repeat(top_embedding, repeats=50257, axis=0)
    # top_embedding = top_embedding.numpy()
    # print(top_embedding)


    # print(cosine_similarity(top_embedding, word_embeddings).shape)
    # cosine_similarity = tf.keras.metrics.CosineSimilarity(axis=-1)
    # cosine_similarity.update_state(top_embedding, word_embeddings)
    # print(cosine_similarity.result().numpy())

    top_s = None
    return top_s

def cal_cos_sim(word, vocab):

    cossim = tf.Variable(tf.zeros(50257))
    m = tf.keras.metrics.CosineSimilarity(axis=1)

    for i, line in tqdm(enumerate(vocab), total=len(vocab)):

        m.update_state(word, line)
        cossim = cossim[i].assign(m.result())
        m.reset_states()




    # print(cossim)


    return cossim.numpy()


if __name__ == "__main__" :

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model_type = 'gpt2'
    prompts = ['These sentences are to test some text generation tasks.',
                'I want a dog.',
               'He wants a dog.',
               ]

    tokenizer, model = prepare_tknzr_model(model_type, framework='torch')
    modeltf = prepare_tknzr_model(model_type, framework='tf')[1]

    # position_embeddings = model.transformer.wpe.weight[:]

    # print(position_embeddings)
    print(torch.cuda.is_available())


    for prompt in prompts:
        print("\nPrompt:\t", prompt)
        input_ids_tf = tokenizer.encode(prompt, return_tensors='tf')
        print("Default Sampling:\t",sample_default(input_ids_tf, tokenizer, modeltf, max_length=100))
        # print("Greedy:\t",greedy(input_ids_tf, tokenizer, modeltf, max_length=100))
        # print("Top S:\t", top_s(input_ids_tf, tokenizer, modeltf))
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        # print("Entmax:\t", entmax(input_ids, tokenizer, model, max_length=100))


