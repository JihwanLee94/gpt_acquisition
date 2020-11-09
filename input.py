from copy import deepcopy
from pprint import pprint

Sentences = {'go':
                 [['I go there.', 'go'],
                  ['I goed there.', 'goed'],
                  ['I goes there.', 'goes'],
                  ['I went there.', 'went'],
                  ['I wented there.', 'wented']],
             'eat':
                 [['I eat this.', 'eat'],
                  ['I eated this.', 'eated'],
                  ['I eats this.', 'eats'],
                  ['I ate this.', 'ate'],
                  ['I ated this.', 'ated']],
             'know':
                 [['I know this.', 'know'],
                  ['I knowed this.', 'knowed'],
                  ['I knows this.', 'knows'],
                  ['I knew this.', 'knew'],
                  ['I knewed this.', 'knewed']],
             'write':
                 [['I write this.', 'write'],
                  ['I writed this.', 'writed'],
                  ['I writes this.', 'writes'],
                  ['I wrote this.', 'wrote'],
                  ['I wroted this.', 'wroted']],
             'feel':
                 [['I feel this.', 'feel'],
                  ['I feeled this.', 'feeled'],
                  ['I feels this.', 'feels'],
                  ['I felt this.', 'felt'],
                  ['I felted this.', 'felted']],

             ######## regular ########
             'like':
                 [['I like this.', 'like'],
                  ['I liked this.', 'liked'],
                  ['I likes this.', 'likes']],

             'hate':
                 [['I hate this.', 'hate'],
                  ['I hated this.', 'hated'],
                  ['I hates this.', 'hates']],

             'want':
                 [['I want this.', 'want'],
                  ['I wanted this.', 'wanted'],
                  ['I wants this.', 'wants']],

             }


Questions_pres = deepcopy(Sentences)
Questions_past = deepcopy(Sentences)

for k in Questions_pres.keys():
    for i, item in enumerate(Questions_pres[k]):
        last = item[0].split()[-1][:-1]
        Questions_pres[k][i] = [f'Do you {k} {last}?\nYes I ', item[0][2:], item[1]]
        Questions_past[k][i] = [f'Did you {k} {last}?\nYes I ', item[0][2:], item[1]]

        # print('item', questions[k][i])
    # print(questions[k])


pprint(Questions_pres)
pprint(Questions_past)
