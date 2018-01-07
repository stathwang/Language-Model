#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import time
from collections import defaultdict
from konlpy.tag import Twitter

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

def read_data(filename):
    with open(filename, 'r') as f:
        corpus = [row.split('\t') for row in f.read().splitlines()]
        corpus = corpus[1:]
    return corpus

def tokenize_tag(corpus):
    pos_tagger = Twitter()
    return ['/'.join(a) for a in pos_tagger.pos(corpus)]

def calc_ngram(sentence, n):
    tokens = [START_SYMBOL] * (n-1) + sentence.strip().split() + [STOP_SYMBOL]
    ngrams = [tuple(tokens[i:(i+n)]) for i in range(len(tokens)-n+1)]
    return ngrams

def calc_probabilities(training_corpus):
    unigram_c = defaultdict(int)
    bigram_c = defaultdict(int)
    trigram_c = defaultdict(int)

    for sent in training_corpus:
        tokens_u = calc_ngram(sent, 1)
        tokens_b = calc_ngram(sent, 2)
        tokens_t = calc_ngram(sent, 3)

        for unigram in tokens_u:
            unigram_c[unigram] += 1

        for bigram in tokens_b:
            bigram_c[bigram] += 1

        for trigram in tokens_t:
            trigram_c[trigram] += 1

    unigram_total = sum(unigram_c.values())
    unigram_p = {a: math.log(unigram_c[a], 2) - math.log(unigram_total, 2) for a in unigram_c}

    unigram_c[START_SYMBOL] = len(training_corpus)
    bigram_p = {(a, b): math.log(bigram_c[(a, b)], 2) - math.log(unigram_c[(a,)], 2) for a, b in bigram_c}

    bigram_c[(START_SYMBOL, START_SYMBOL)] = len(training_corpus)
    trigram_p = {(a, b, c): math.log(trigram_c[(a, b, c)], 2) - math.log(bigram_c[(a, b)], 2) for a, b, c in trigram_c}

    return unigram_p, bigram_p, trigram_p

def q1_output(unigrams, bigrams, trigrams, filename):
    outfile = open(filename, 'w')

    unigrams_keys = list(unigrams.keys())
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = list(bigrams.keys())
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1] + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = list(trigrams.keys())
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()

def score(ngram_p, n, corpus):
    scores = []
    for sent in corpus:
        ngrams = calc_ngram(sent, n)
        prob_sent = 0
        if all(ngram in ngram_p for ngram in ngrams):
            for ngram in ngrams:
                prob_sent += ngram_p[ngram]
            scores.append(prob_sent)
        else:
            scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
    return scores

def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    global_lambda = 1.0 / 3
    for sent in corpus:
        ngrams = calc_ngram(sent, 3)
        lin_interpolated_score = 0
        for trigram in ngrams:
            p3 = trigrams.get(trigram, MINUS_INFINITY_SENTENCE_LOG_PROB)
            p2 = bigrams.get(trigram[1:3], MINUS_INFINITY_SENTENCE_LOG_PROB)
            p1 = unigrams.get(trigram[2], MINUS_INFINITY_SENTENCE_LOG_PROB)
            lin_interpolated_score += math.log(global_lambda * (2**p3 + 2**p2 + 2**p1), 2)
        scores.append(lin_interpolated_score)
    return scores

DATA_PATH = 'PATH_TO_YOUR_DATA_FILES'
OUTPUT_PATH = 'PATH_TO_YOUR_OUTPUT'

def main():
    time.clock()

    infile = DATA_PATH + 'ratings.txt'
    train = read_data(infile)
    doc = [tokenize_tag(row[1]) for row in train]

    corpus = []
    for tag_token in doc:
        sent = ' '.join([elem.split('/')[0] for elem in tag_token])
        corpus.append(sent)

    outfile = open(OUTPUT_PATH + 'corpus.txt', 'w')
    for row in corpus:
        outfile.write(row + '\n')
    outfile.close()

    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'ngram_probs.txt')

    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    score_output(uniscores, OUTPUT_PATH + 'unigram_scores.txt')
    score_output(biscores, OUTPUT_PATH + 'bigram_scores.txt')
    score_output(triscores, OUTPUT_PATH + 'trigram_scores.txt')

    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    score_output(linearscores, OUTPUT_PATH + 'linear_scores.txt')

    print('Time: ' + str(time.clock()) + ' sec')

if __name__=='__main__':
    main()
