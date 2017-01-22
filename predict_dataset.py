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

def predict(model, dataset, iteration=1):
    ans = []
    for index, item in enumerate(dataset):
        current_max = -100000.0
        current_ans = None
        for i in range(0, iteration):
            result = model.predict(np.array([dataset.get_example(index)[0]])).data
            if np.max(result) > current_max:
                current_max = np.max(result)
                current_ans = np.argmax(result)

        print(str(index) + ' ' + str(current_ans))
        ans.append(current_ans)
    return ans

def output_submit_file(ans, output_filename):
    f = open(output_filename, 'w')
    for index, row in enumerate(ans):
        f.write("{0},{1}".format(index,ans[index]))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model_snapshot', '-m', default=None, help='Filename of model snapshot')
    parser.add_argument('--net', '-n', default='GoogLeNet', help='Choose network to use for prediction')
    parser.add_argument('--iteration', '-t', type=int, default=1, help='Sampling iteration for each test data')
    parser.add_argument('--output', '-o', default='default', help='Sampling iteration for each test data')
    args = parser.parse_args()

    if args.output == 'default':
        output_filename = 'output_' + args.model_snapshot + '_' + str(args.iteration) + '.csv'

    model = ''
    if args.net == 'CNN':
        model = CNN()
    elif args.net == 'GoogLeNet':
        model = GoogLeNetBN()
    else:
        print('please select CNN or GoogLeNet')
    chainer.serializers.load_npz(args.model_snapshot, model)
    trial_data = ImageDataset(normalize=True, flatten=False, max_size=224, dataselect=-1, mode='trial')
    #trial_data = ImageDataset(normalize=True, flatten=False, max_size=224, dataselect=-1, mode='train')
    ans = predict(model, trial_data, args.iteration)
    output_submit_file(ans, output_filename)
