#! /usr/bin/env python
# coding: utf-8
# coding=utf-8
# -*- coding: utf-8 -*-
# vim: fileencoding=utf-8
import sys
import random
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import csv
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class ImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, normalize=True, flatten=True, train=True, max_size=200):
        self._normalize = normalize
        self._flatten = flatten
        self._train = train
        self._max_size = max_size
        pairs = []
        with open('data/clf_train_master.tsv', newline='') as f:
            tsv = csv.reader(f, delimiter='\t')
            for row in tsv:
                if 'jpg' in row[0]:
                    pairs.append(row)

        im = self.get_image(pairs[4][0])
        self._pairs = pairs

    def __len__(self):
        return len(self._pairs)

    def get_image(self, filename):
        image = Image.open('data/' + filename)
        #new_w = int(image.size[0] * (self._max_size / max(image.size)))
        #new_h = int(image.size[1] * (self._max_size / max(image.size)))
        new_w = self._max_size
        new_h = self._max_size
        image = image.resize((new_w, new_h), Image.BICUBIC)
        image_array = np.asarray(image)
        return image_array
        
        # type cast
        image_array = image_array.astype('float32')
        label = np.int32(label)
        return image_array, label

    def get_example(self, i):
        filename = self._pairs[i][0]
        image_array = self.get_image(filename)
        if self._normalize:
            image_array = image_array / np.max(image_array)
        if self._flatten:
            image_array = image_array.flatten()
        else:
            if image_array.ndim == 2:
                mage_array = image_array[np.newaxis,:]

        image_array = image_array.astype('float32')
        image_array = image_array.transpose(2, 0, 1) # order of rgb / h / w
        label = np.int32(self._pairs[i][1])
        return image_array, label
