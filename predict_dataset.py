#! /usr/bin/env python
# coding: utf-8
# coding=utf-8
# -*- coding: utf-8 -*-
# vim: fileencoding=utf-8
import sys
import random
import numpy as np
from PIL import Image
import csv
import chainer
import chainer.links as L
from chainer import datasets
from train import *
from image_dataset import *

def predict(model, dataset):
    ans = []
    for index, item in enumerate(dataset):
        ans.append(np.argmax(model(np.array([dataset.get_example(index)[0]])).data))

    return ans

def print_submit(ans):
    for index, row in enumerate(ans):
        print("{0},{1}".format(index,ans[index]))
    print("")

if __name__ == '__main__':
    model = CNN()
    chainer.serializers.load_npz('cnn_epoch_20.npz', model)
    train_data = ImageDataset(normalize=True, flatten=False, train=True, max_size=200, dataselect=list(range(0,20)))
    ans = predict(model, train_data)
    print_submit(ans)
