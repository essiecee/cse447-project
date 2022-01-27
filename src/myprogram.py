#!/usr/bin/env python
import os
import json
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


def get_pred(context_input, context_dict, num_preds=3):
    """
    return 3 distinct word predictions given context_input
    """
    # we can assume there is at least 3 possible next words
    # given context, if we use enough data
    assert len(context_dict[context_input]) >= num_preds
    # TODO: need to consider if context_input doesn't exist in context_dict

    # convert all nested dicts to FreqDist so that we can use library utils like most_common()
    for k in context_dict:
        fdist = FreqDist(context_dict[k])
        context_dict[k] = fdist

    # sorts frequency of each word given context in descending order
    topk_preds = context_dict[context_input].most_common()
    pred_list = [None] * num_preds
    # ensure distinct word predictions
    ignore_words = set()

    for i in range(num_preds):
        prob_sum = 0
        rand_prob = random.random()
        for pred in topk_preds:
            count = topk_preds[pred]
            prob = count / (context_dict[context_input]).N()  # topk_preds.N()
            prob_sum += prob

            if prob_sum > rand_prob and pred not in ignore_words:
                pred_list[i] = pred
                ignore_words.add(pred)
                break

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
            self.bigrams_context_freq = n_grams_models['bigrams']
            self.trigrams_context_freq = n_grams_models['trigrams']

        # if a word appears less than x number of times, then it should be replaced by an UNK
        self.UNK_THRESHOLD = 2
        # token: "UNK" or token, depending on whether the token should be UNK-ed or not
        # probably don't need this when reloading the model? o.w. we should save it to file too
        self.word_mappings = {}

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        dataset = load_dataset('csebuetnlp/xlsum', 'english', split='train')
        sentences = []
        # TODO: set this to be dataset.num_rows when we want to train on a larger set of data
        num_rows = 2  # dataset.num_rows
        for i in range(num_rows):
            sentences.append(dataset[i]['text'])
        # TODO: what should this function return? currently it returns a list of sentences (just two for now)
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
        for paragraph in data:
            # tokenize text and build contexts for n_gram models.
            build_contexts(paragraph, self.unigrams_context_freq, n=1)

            # determine which tokens should be UNK-ed
            for k, v in self.unigrams_context_freq['()'].items():
                if v < self.UNK_THRESHOLD:
                    self.word_mappings[k] = "UNK"
                else:
                    self.word_mappings[k] = k

            build_contexts(paragraph, self.bigrams_context_freq, n=2, word_mappings=self.word_mappings)
            build_contexts(paragraph, self.trigrams_context_freq, n=3, word_mappings=self.word_mappings)

    def populate_frequency_maps(self, ngrams, map):
        for token in ngrams:
            if token not in map:
                map[token] = 1
            else:
                map[token] += 1

    def run_pred(self, data):
        # your code here
        unigram_preds = []
        bigram_preds = []
        trigram_preds = []
        for seq in data:
            seq_list = seq.split()  # example: ['Happy', 'New', 'Yea']
            # this way, we will have full words only in our context
            # we can then filter the predicted words for ones that start with the incomplete word's letters
            context_list = seq_list[:-1]  # example: ['Happy', 'New']
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

            # print("bigram_context = " + str(bigram_context))
            # print("trigram_context = " + str(trigram_context))

            unigram_top_guesses = get_pred(unigram_context, self.unigrams_context_freq)
            unigram_preds.append(''.join(unigram_top_guesses))
            bigram_top_guesses = get_pred(bigram_context, self.bigrams_context_freq)
            bigram_preds.append(''.join(bigram_top_guesses))
            trigram_top_guesses = get_pred(trigram_context, self.trigrams_context_freq)
            trigram_preds.append(''.join(trigram_top_guesses))

        return unigram_preds, bigram_preds, trigram_preds

        # # starter code for reference
        # preds = []
        # all_chars = string.ascii_letters
        # for inp in data:
        #     # this model just predicts a random character each time
        #     top_guesses = [random.choice(all_chars) for _ in range(3)]
        #     preds.append(''.join(top_guesses))
        # return preds

    def save(self, work_dir):
        n_grams_models = {
            'unigrams': self.unigrams_context_freq,
            'bigrams': self.bigrams_context_freq,
            'trigrams': self.trigrams_context_freq
        }
        with open(os.path.join(work_dir, 'model.checkpoint'), 'w') as output_json:
            json.dump(n_grams_models, output_json)

        # # your code here
        # # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
        #     f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        path = Path(os.path.join(work_dir, 'model.checkpoint'))
        with open(path, 'rb') as f:
            data = json.load(f)

        # convert string keys back to tuples "('a', 'b')" -> ('a', 'b')
        data = {ngram: {literal_eval(k): v for k, v in data[ngram].items()} for ngram in data}
        return MyModel(n_grams_models=data)

        # # your code here
        # # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint')) as f:
        #     dummy_save = f.read()
        # return MyModel()


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
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
