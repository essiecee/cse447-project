#!/usr/bin/env python
import os
import json
import string
import random
from pathlib import Path
from ast import literal_eval
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk import FreqDist
from typing import Dict
from collections import defaultdict
from tqdm import tqdm
import time


def build_contexts(text, context_dict: Dict[tuple, Dict[str, int]], n, word_mappings=None):
    """
    build n_grams given n
    assigns each word x_t and their respective counts to their context x_{t-n+1:t-1}
    """
    tokens = ['<s>'] * (n - 1)
    # UNK text as needed
    tokenized_text = word_tokenize(text)

    if word_mappings:  # in the unigram model we shouldn't use word_mappings since it hasn't been constructed yet
        tokenized_text = [word_mappings[word] for word in tokenized_text]
    tokens.extend(tokenized_text)
    tokens.append('</s>')
    n_grams = ngrams(tokens, n)


    for ngram in n_grams:
        context = ngram[:-1]
        target = ngram[-1]
        # convert tuple keys to strings for json
        context_dict[str(context)][target] += 1


def get_pred(context_input, context_dict, target_word, num_preds=3):
    """
    return 3 distinct word predictions given context_input
    """
    # case that context_input doesn't exist in the context_dict
    # returns 3 letters that are random in that case
    if context_input not in context_dict:
        random_pred = []
        for i in range(3):
            random_pred.append(random.choice(string.ascii_letters))
        return random_pred

    # case where not enough target words for the given context
    # inject random target words into the context_dict until there's enough for 3 predictions
    # if there's any key, then there's guaranteed to be at least one target word value in the dictionary
    while True:
        try:
            assert len(context_dict[context_input]) >= num_preds
            break
        except AssertionError:
            context_dict[context_input][random.choice(string.ascii_letters)] = 1

    # convert all nested dicts to FreqDist so that we can use library utils like most_common()
    # for k in context_dict: # potential bottleneck???
    #     fdist = FreqDist(context_dict[k])
    #     context_dict[k] = fdist

    # sorts frequency of each word given context in descending order
    topk_preds = context_dict[context_input] #.most_common()
    pred_list = []
    # ensure distinct word predictions
    ignore_words = set()


    for pred in topk_preds:

        # potential word that may match with target
        word = pred[0]    # Ex. target_word is "on" and the word is "one"
        """
        count = pred[1]
        prob = count / (context_dict[context_input]).N()  # topk_preds.N()
        prob_sum += prob
        if prob_sum > rand_prob and word not in ignore_words:
            pred_list[i] = word
            ignore_words.add(word)
            break
        """
        pred_letter = ""

        # checking if the potential word matches up with the target word
        # also checks that the potential word is longer so that there are characters we can pull from
        if word.startswith(target_word) and len(target_word) < len(word):

            pred_letter = word[len(target_word)]

            # to increase our chance of guessing the right character, we will look for other
            # characters and ignore the repeat characters
            if pred_letter not in ignore_words:
                pred_list.append(pred_letter)
                if len(pred_list) == num_preds:
                    break
                ignore_words.add(pred_letter)

    # checking that we have 3 letters for sure as our prediction
    while True:
        try:
            assert len(pred_list) == num_preds
            break
        except AssertionError:
            pred_list.append(random.choice(string.ascii_letters))

    return pred_list


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self, n_grams_models=None):
        if n_grams_models is None:
            self.unigrams_context_freq = defaultdict(FreqDist)
            self.bigrams_context_freq = defaultdict(FreqDist)
            self.trigrams_context_freq = defaultdict(FreqDist)
        else:
            self.unigrams_context_freq = n_grams_models['unigrams']
            # self.bigrams_context_freq = n_grams_models['bigrams']
            # self.trigrams_context_freq = n_grams_models['trigrams']

        # if a word appears less than x number of times, then it should be replaced by an UNK
        self.UNK_THRESHOLD = 2
        # token: "UNK" or token, depending on whether the token should be UNK-ed or not
        self.word_mappings = {}

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        # adding more languages other than English to be a little more robust/generalizable
        top_languages = ['english', 'spanish', 'chinese_simplified', 'hindi', 'arabic']
        lang_datasets = [load_dataset('csebuetnlp/xlsum', lang, split='train') for lang in top_languages]

        sentences = []
        # TODO: set this to be dataset.num_rows when we want to train on a larger set of data
        num_rows = 2000  # 2
        for dataset in lang_datasets:
            for i in range(num_rows):
                sentences.append(dataset[i]['text'])
        return sentences

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        for i, paragraph in enumerate(data):
            if i % 1000 == 0:
                print('training instance:', i, ' / ', len(data))
            # tokenize text and build contexts for n_gram models.
            build_contexts(paragraph, self.unigrams_context_freq, n=1)

            # determine which tokens should be UNK-ed
            for k, v in self.unigrams_context_freq['()'].items():
                if v < self.UNK_THRESHOLD:
                    self.word_mappings[k] = "UNK"
                else:
                    self.word_mappings[k] = k

            # build_contexts(paragraph, self.bigrams_context_freq, n=2, word_mappings=self.word_mappings)
            # build_contexts(paragraph, self.trigrams_context_freq, n=3, word_mappings=self.word_mappings)


    def populate_frequency_maps(self, ngrams, map):
        for token in ngrams:
            if token not in map:
                map[token] = 1
            else:
                map[token] += 1

    def run_pred(self, data):
        # your code here
        unigram_preds = []
        # bigram_preds = []
        # trigram_preds = []
        for seq in tqdm(data):
            seq_list = seq.split()  # example: ['Happy', 'New', 'Yea']
            # this way, we will have full words only in our context
            # we can then filter the predicted words for ones that start with the incomplete word's letters
            context_list = seq_list[:-1]  # example: ['Happy', 'New']

            # mapping all the words that we haven't seen yet to UNK
            for i in range(len(context_list)):
                word = context_list[i]
                if word in self.word_mappings:
                    word = self.word_mappings[word]
                else:   # not in the mapping, set to UNK
                    word = "UNK"
                context_list[i] = word

            unigram_context = ()
            # add start symbols for the contexts
            if len(seq_list) == 1:
                bigram_context = ('<s>',)
            else:
                bigram_context = tuple(context_list[-1:])
            if len(seq_list) == 1:
                trigram_context = ('<s>', '<s>')
            elif len(seq_list) == 2:
                trigram_context = ('<s>', context_list[-1])
            else:
                trigram_context = tuple(context_list[-2:])

            unigram_top_guesses = get_pred(unigram_context, self.unigrams_context_freq, seq_list[len(seq_list) - 1])
            unigram_preds.append(''.join(unigram_top_guesses))
            # bigram_top_guesses = get_pred(bigram_context, self.bigrams_context_freq, seq_list[len(seq_list) - 1])
            # bigram_preds.append(''.join(bigram_top_guesses))
            # trigram_top_guesses = get_pred(trigram_context, self.trigrams_context_freq, seq_list[len(seq_list) - 1])
            # trigram_preds.append(''.join(trigram_top_guesses))

        return unigram_preds #, bigram_preds, trigram_preds

    def save(self, work_dir):
        start = time.perf_counter()
        for context in self.unigrams_context_freq:
            descending = self.unigrams_context_freq[context].most_common()
            self.unigrams_context_freq[context] = descending[:(len(descending) // 2)]

        # for context in self.bigrams_context_freq:
        #     descending = self.bigrams_context_freq[context].most_common()
        #     self.bigrams_context_freq[context] = descending

        # for context in self.trigrams_context_freq:
        #     descending = self.trigrams_context_freq[context].most_common()
        #     self.trigrams_context_freq[context] = descending

        end = time.perf_counter()
        # print("time it took to convert: " + str(end - start))
        n_grams_models = {
            'unigrams': self.unigrams_context_freq
            # 'unigrams': self.unigrams_context_freq,
            # 'bigrams': self.bigrams_context_freq,
            # 'trigrams': self.trigrams_context_freq
        }
        with open(os.path.join(work_dir, 'model.checkpoint'), 'w') as output_json:
            json.dump(n_grams_models, output_json)

    @classmethod
    def load(cls, work_dir):
        path = Path(os.path.join(work_dir, 'model.checkpoint'))
        with open(path, 'rb') as f:
            data = json.load(f)

        # convert string keys back to tuples "('a', 'b')" -> ('a', 'b')
        data = {ngram: {literal_eval(k): v for k, v in data[ngram].items()} for ngram in data}
        return MyModel(n_grams_models=data)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':    # chose unigram model as it gave us the best results during testing
        print('Loading model')
        start_time = time.perf_counter()
        model = MyModel.load(args.work_dir)
        end_time = time.perf_counter()
        print("model loading took: " + str(end_time - start_time))
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        # unigram_pred, _, trigram_pred = model.run_pred(test_data)
        unigram_pred = model.run_pred(test_data)    
        print('Writing predictions to {}'.format(args.test_output))
        # currently using unigram predictions
        assert len(unigram_pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(unigram_pred))
        model.write_pred(unigram_pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
