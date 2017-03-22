# _*_ coding=utf-8 -*_
'''
Created on 2017-03-21
@author: yyf
'''

import codecs
import os
from tqdm import tqdm
import random


class WordDict:
    '''
    word-count key-value dictionary
    key is a utf-8 string representing a word or a N-gram
    '''
    def __init__(self):
        self.dict = {}

    def add_ngram(self, key):
        if self.dict.has_key(key):
            self.dict[key] += 1
        else:
            self.dict[key] = 1

    def calc_freq(self):
        self.freq = self.dict.items()
        self.freq = sorted(self.freq, key=lambda a: -a[1])

    def filter(self, ngrams):
        '''
        Remove ngrams the may be not useful
        '''
        for k in ngrams:
            if self.dict.has_key(k):
                self.dict.pop(k)

    def arrange(self):
        self.calc_freq()
        self.ind2ngram = self.dict.keys()
        self.ngram2ind = {}
        for idx,ng in enumerate(self.ind2ngram):
            self.ngram2ind[ng] = idx


def deal_passage(file, use_word_gram=False):
    '''
    :param file: path name
    :return: a string with blanks removed.
    '''
    passage = []
    with codecs.open(file, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            passage.extend(line.split())
            if use_word_gram:
                passage.extend([w for w in "".join(line.split())])
    return passage


class DataSet:
    '''
    class_label : all passages processed.
    '''
    def __init__(self):
        # label and passages
        self.data = {}
        # (label, passage) pairs

    def add(self, label, passages):
        self.data[label] = passages

    def __shuffle__(self):
        self.input = []
        self.labels = []
        for lbl in self.data.keys():
            self.labels.append(lbl)
            for psg in self.data[lbl]:
                self.input.append((lbl, psg))
        random.shuffle(self.input)

    def partition(self, nsets=10):
        self.__shuffle__()
        step = int(len(self.input) / nsets)
        # print("dataset partition wit step %d, last step %d " % (step, len(self.input) % step))
        for begin in range(0, len(self.input), step):
            right = begin + min(len(self.input) - begin, step)
            if right - begin < 1000:
                continue
            test = self.input[begin: right]
            left = self.input[0:begin]
            right = self.input[right: right+step]
            left.extend(right)
            yield (test, left)


def read_data(datadir):
    dataset = DataSet()
    for dir in tqdm(os.listdir(datadir)):
        label = dir.decode(encoding="utf-8")
        # print (u"process class: " + label)
        passages = []
        for file in os.listdir(os.path.join(datadir, dir)):
            passages.append(deal_passage(os.path.join(datadir, dir, file)))
        dataset.add(label, passages)
    return dataset



