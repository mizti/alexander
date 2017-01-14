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
from chainer import datasets

class ImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, normalize=True, flatten=True, train=True, max_size=200, datasize = 0, data_dir='data'):
        self._normalize = normalize
        self._flatten = flatten
        self._train = train
        self._max_size = max_size
        self._data_dir = data_dir
        pairs = []
        with open(self._data_dir + '/clf_train_master.tsv', newline='') as f:
            tsv = csv.reader(f, delimiter='\t')
            for row in tsv:
                if 'jpg' in row[0]:
                    pairs.append(row)
        if datasize > 0:
            pairs = random.sample(pairs, datasize)
        self._pairs = pairs

    def __len__(self):
        return len(self._pairs)

    def get_image(self, filename):
        image = Image.open(self._data_dir + '/' + filename)
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
            image_array = image_array / 255
        if self._flatten:
            image_array = image_array.flatten()
        else:
            if image_array.ndim == 2:
                mage_array = image_array[np.newaxis,:]
        image_array = image_array.astype('float32')
        image_array = image_array.transpose(2, 0, 1) # order of rgb / h / w
        label = np.int32(self._pairs[i][1])
        return image_array, label
