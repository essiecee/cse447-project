#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
import nltk
from nltk import word_tokenize
from nltk.util import ngrams

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    unigrams_count = {}
    bigrams_count = {}
    trigrams_count = {}

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        dataset = load_dataset('csebuetnlp/xlsum', 'english', split='train')
        sentences = []
        # TODO: set this to be dataset.num_rows when we want to train on a larger set of data
        num_rows = 2 # dataset.num_rows
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
            unigrams = ngrams(paragraph.split(), 1)
            # each token in unigrams looks like ('word',)
            # consider doing extra parsing for the raw text, e.g. token[0]
            self.populate_frequency_maps(unigrams, MyModel.unigrams_count)

            bigrams = ngrams(paragraph.split(), 2)
            self.populate_frequency_maps(bigrams, MyModel.bigrams_count)

            trigrams = ngrams(paragraph.split(), 3)
            self.populate_frequency_maps(trigrams, MyModel.trigrams_count)

    def populate_frequency_maps(self, ngrams, map):
        for token in ngrams:
            if token not in map:
                map[token] = 1
            else:
                map[token] += 1

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


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
