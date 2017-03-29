# _*_ coding=utf-8 -*_
'''
Created on 2017-03-21
@author: yyf
'''

import random
import numpy as np
from tqdm import tqdm

from tools import *
from naivebayes import *
from simple_softmax import *

DATADIR = "/home/yyf/Documents/ml/introduction2ml/new_weibo_13638"
USE_WORD_GRAM = False
DELTA = 0.7
USENVIVEBAYES = True
USESOFTMAX = False
GRADIENT_CHECK = False
USEBOOTSTRAP = False

# dirs = os.listdir(DATADIR)
# for dir in dirs:
#     print(dir.decode(encoding="utf-8"))
#     files = os.listdir(os.path.join(DATADIR, dir))
#     print(len(files))
#     print(deal_passage(os.path.join(DATADIR, dir, files[1])))
#     break

dataset = read_data(DATADIR)

if USENVIVEBAYES:
    precisions = []
    for d in range(1, 11, 2):
        topk = [1, 2]
        tpre = []
        for test,train in dataset.partition():
            pre = []
            nb = NaiveBayes(test, train, dataset.labels, d / 10.)
            nb.train()
            for k in topk:
                pre.append(nb.precision(k))
            # pre.append(nb.train_precision())
            tpre.append(pre)
        precisions.append(np.mean(tpre, axis=0))

    print ("Final results:")
    print(precisions)

if GRADIENT_CHECK:

    model = Softmax(False, 2, 2)
    model.regular = 0.1
    model.use_regular = True
    X = np.random.random_sample((2, 1))
    Y = np.random.random_sample((2, 1))
    Y = model.__softmax__(Y)
    model.gradient_check(X, Y)

    print ("sparsed version")
    model = Softmax(True, 100, 10)
    X = np.array([[1, 9]])
    Y = np.random.random_sample((10, 1))
    Y = model.__softmax__(Y)
    model.gradient_check(X, Y)

    print("check y")
    model = Softmax(True, 100, 10)
    model.regular = 0.2
    model.use_regular = True
    X = np.array([[1, 9]])
    Y = np.array([5])
    model.gradient_check(X, Y)

def bootstrap(X, y, k=3):
    bx, by = [], []
    for idx, x in enumerate(X):
        x = np.array(x)
        indices = np.arange(len(x))
        for _ in range(k):
            random.shuffle(indices)
            bx.append(x[indices % k != 0])
            by.append(y[idx])
    return bx, by

if USESOFTMAX:
    word_dict = dataset.word_dict()
    for test, train in dataset.partition():
        idx2label = dataset.labels
        label2idx = {}
        for idx, lbl in enumerate(idx2label):
            label2idx[lbl] = idx
        num_features = len(word_dict.ngram2ind)
        num_classes = len(idx2label)
        n = len(train)
        model = Softmax(True, num_features, 100, num_classes)
        print(num_features, num_classes, n)
        print (len(dataset.input))
        print(len(test))
        print(len(train))
        tx, ty = [], []
        for lbl, psg in test:
            ty.append(label2idx[lbl])
            tx.append([])
            for ng in psg:
                tx[-1].append(word_dict.ngram2ind[ng])
        trainx, trainy = [], []
        for lbl, psg in train:
            trainx.append([])
            trainy.append(label2idx[lbl])
            for ng in psg:
                trainx[-1].append(word_dict.ngram2ind[ng])
        if USEBOOTSTRAP:
            bx, by = bootstrap(trainx, trainy)
            trainx.extend(bx)
            trainy.extend(by)
            print(len(trainx))
        model.fit(trainx, trainy)
        for run in range(30):
            print("\nrun %d:"%(run))
            if model.train(max_epoch=30, eval_epoch=10) == 0:
                print("test precision is %f " %(model.precision(tx, ty)))
            print (model.precision_log)
            print("test precision is %f " %(model.precision(tx, ty)))
        break






