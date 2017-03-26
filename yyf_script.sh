#!/bin/bash


BAYESDIR=/home/yyf/Documents/codes/pycharm/ML/
GITDIR=/home/yyf/Documents/ml/introduction2ml/weibo


export PYTHONPATH=$PYTHONPATH:$BAYESDIR:$BAYESDIR/weibo

rsync -auv $BAYESDIR/weibo/* $GITDIR/yyf

git add --all
git commit -am "$(date)"
git push


