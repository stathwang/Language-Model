#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'

def partial_match(key, ngram_dict):
    for k, v in ngram_dict.items():
        if all(k1 == k2 or k2 is None for k1, k2 in zip(k, key)):
            yield k, v

def weighted_pick(words, ngram_dict, n):
    partial_dict = dict(partial_match(words, ngram_dict))
    next_words = []
    probs = []
    items = partial_dict.items()
    for trigram, log_prob in items:
        next_words.append(trigram[-1])
        probs.append(2 ** float(log_prob))
    norm_probs = [a / sum(probs) for a in probs]
    key = np.random.choice(next_words, 1, p=norm_probs)
    return str(key[0])

def generate_sentence(ngram_dict, n):
    sentence = []
    words = [START_SYMBOL] * (n-1) + [None]

    word = weighted_pick(words, ngram_dict, n)
    while word != STOP_SYMBOL:
        if n != 1:
            del words[0]
            del words[-1]
            words.append(word)
            words.append(None)
        sentence.append(word)
        word = weighted_pick(words, ngram_dict, n)

    print(' '.join(sentence))

OUTPUT_PATH = 'PATH_TO_YOUR_OUTPUT'

def main():
    with open(OUTPUT_PATH + 'ngram_probs.txt', 'r') as f:
        ngrams = [row.split(' ') for row in f.read().splitlines()]
        trigram_list = [row[1:] for row in ngrams if row[0] == 'TRIGRAM']
        trigram_dict = defaultdict()
        for a1, a2, a3, b in trigram_list:
            trigram_dict[(a1, a2, a3)] = float(b)

    print('Generating random sentences...')
    for i in range(20):
        generate_sentence(trigram_dict, 3)

if __name__=='__main__':
    main()
