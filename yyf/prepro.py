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

DATADIR = "/home/yyf/Documents/ml/introduction2ml/new_weibo_13638"
USE_WORD_GRAM = False
DELTA = 0.7

# dirs = os.listdir(DATADIR)
# for dir in dirs:
#     print(dir.decode(encoding="utf-8"))
#     files = os.listdir(os.path.join(DATADIR, dir))
#     print(len(files))
#     print(deal_passage(os.path.join(DATADIR, dir, files[1])))
#     break

dataset = read_data(DATADIR)

precisions = []
for d in tqdm(range(1, 11)):
    topk = [1, 3, 5]
    pre = []
    for test,train in tqdm(dataset.partition()):
        nb = NaiveBayes(test, train, dataset.labels, d / 10.)
        nb.train()
        for k in topk:
            pre.append(nb.precision(k))
        pre.append(nb.train_precision())
    precisions.append(np.mean(pre, axis=0))

print ("Final results:")
print(precisions)
