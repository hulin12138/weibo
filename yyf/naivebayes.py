# _*_ coding=utf-8 -*_
'''
Created on 2017-03-21
@author: yyf
'''

import numpy as np

from tools import *

NGRAM_TO_REMOVE = u""

class NaiveBayes:

    def __init__(self, test, train, labels, delta=1.):
        self.data_test = test
        self.data_train = train
        self.ind2label = labels
        self.label2ind = {}
        self.word_dict = WordDict()
        self.delta = delta
        for no,lbl in enumerate(labels):
            self.label2ind[lbl] = no

    def smooth(self, delta=1.0):
        '''
        Add delta smoothing.
        '''
        self.condition_probability += delta

    def train(self):
        self.num_classes = len(self.ind2label)
        for lbl,psg in self.data_train:
            for w in psg:
                self.word_dict.add_ngram(w)
        self.word_dict.filter(NGRAM_TO_REMOVE)
        self.word_dict.arrange()
        self.num_ngram = len(self.word_dict.freq)
        self.condition_probability = np.zeros(shape=(self.num_ngram, self.num_classes), dtype=np.float64)
        for lbl, psg in self.data_train:
            cidx = self.label2ind[lbl]
            for w in psg:
                if self.word_dict.dict.has_key(w):
                    widx = self.word_dict.ngram2ind[w]
                    self.condition_probability[widx][cidx] += 1
        self.smooth(self.delta)
        self.condition_probability /= np.sum(self.condition_probability, axis=0)
        assert(abs(np.sum(self.condition_probability) - self.num_classes) < 1e-5)

    def predict(self, passage, all_prob=False):
        '''
        If all_prob is True, then return the probabilities predicted.
        Else return a label(string).
        '''
        post = np.zeros(shape=(self.num_classes), dtype=np.float64)
        for w in passage:
            if self.word_dict.dict.has_key(w):
                widx = self.word_dict.ngram2ind[w]
                for ci in range(self.num_classes):
                    post[ci] += np.log(self.condition_probability[widx][ci])
        if all_prob:
            return post
        return self.ind2label[np.argmax(post)]

    def precision(self, topk=1):
        pre = 0.
        for lbl, psg in self.data_test:
            lbl = self.label2ind[lbl]
            post = np.array(self.predict(psg, True))
            post = post > post[lbl]
            pre += np.sum(post) < topk
        return pre / len(self.data_test)

    def train_precision(self):
        tmp = self.data_test
        self.data_test = self.data_train
        pre = self.precision()
        self.data_test = tmp
        return pre
