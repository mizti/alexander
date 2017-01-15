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
import glob
import re

# dataselect: can be designated with int or list.
# mode: 'train' / 'trial'
class ImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, normalize=True, flatten=True, max_size=200, dataselect = 0, data_dir='data', mode='train'):
        self._normalize = normalize
        self._flatten = flatten
        self._max_size = max_size
        self._data_dir = data_dir
        self._dataselect = dataselect
        self._mode = mode
        pairs = []
        if self._mode == 'train':
            with open(self._data_dir + '/clf_train_master.tsv', newline='') as f:
                tsv = csv.reader(f, delimiter='\t')
                for index, row in enumerate(tsv):
                    if 'jpg' in row[0]:
                        pairs.append(row)
        elif self._mode == 'trial':
            filelist = glob.glob('data/test*.jpg')
            files = [r.split('/')[-1] for r in glob.glob('data/test*.jpg')]
            sorted_files = sorted(files, key=lambda x: self.getnum_from_string(x))
            for item in sorted_files:
                pairs.append([item, 0]) # filename with dummy label

        else:
            print("pass 'train' or 'trial'")  
    
        if (dataselect.__class__ != int) or (dataselect >0):
            pairs = self.select_data(pairs)

        self._pairs = pairs

    def getnum_from_string(self, filename):
        r = re.compile("([0-9]+)")
        m = r.search(filename)
        return int(m.group(1))
    
    def select_data(self, pairs):
        print(self._dataselect)
        if self._dataselect.__class__ == int:
            pairs = random.sample(pairs, self._dataselect)
        elif self._dataselect.__class__ == list:
            new_pairs = []
            for index, item in enumerate(pairs):
                if (index-1 in self._dataselect): 
                    new_pairs.append(item)
            pairs = new_pairs
        return pairs

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
