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

def predict(model, dataset):
    return 'this!'

if __name__ == '__main__':
    model = L.Classifier(CNN())
    chainer.serializers.load_npz('model_epoch379.npz', model)
    print(model)
