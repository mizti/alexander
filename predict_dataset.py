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
from chainer import datasets, iterators, cuda
from train import *
from image_dataset import *
import math

#def predict(model, dataset, predict_iteration=1, minibatch_size=50):
def predict(model, dataset, predict_iteration=1, minibatch_size=3, device=-1):
    ans = []
    iterator = iterators.SerialIterator(dataset, batch_size=minibatch_size, repeat=True, shuffle=False)
    iter_num = math.ceil(len(dataset) / minibatch_size)
    current_ans = []
    for pi in range(predict_iteration):
        print("current predict_iteration=" + str(pi))
        for i in range(iter_num):
            print(i)
            part_dataset = iterator.next()
            xs = []
            for index, item in enumerate(part_dataset):
                xs.append(item[0])
            if device < 0:
            	xs = np.array(xs)
            else: 
                xs = cuda.to_gpu(np.array(xs))
            result = model.predict(xs)
            for index, item in enumerate(result):
                if pi == 0:
                    current_ans.append([np.argmax(item), item[np.argmax(item)]])
                    current_ans = current_ans[0:len(dataset)]
                else:
                    if current_ans[index][1] < item[np.argmax(item)]:
                        current_ans[index][0] = np.argmax(item)
                        current_ans[index][1] = item[np.argmax(item)]
    ans = current_ans
    return ans

def output_submit_file(ans, output_filename, dataset):
    f = open(output_filename, 'w')
    for index, row in enumerate(ans):
        f.write("{0},{1}\n".format(str(index),str(ans[index][0])))
    f.close()

    f2 = open('result/sample.csv', 'w')
    for index, row in enumerate(ans):
        f2.write("{0}\t{1}\r\n".format(dataset.get_filename(index),str(ans[index][0])))
    f2.close()

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
    else:
        output_filename = args.output

    model = ''
    if args.net == 'CNN':
        model = CNN()
    elif args.net == 'GoogLeNet':
        model = GoogLeNetBN()
    else:
        print('please select CNN or GoogLeNet')

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    chainer.serializers.load_npz(args.model_snapshot, model)
    trial_data = ImageDataset(normalize=True, flatten=False, max_size=224, dataselect=-1, mode='trial')
    #trial_data = ImageDataset(normalize=True, flatten=False, max_size=224, dataselect=list([0,1,2,3]), mode='trial')
    #trial_data = ImageDataset(normalize=True, flatten=False, max_size=224, dataselect=-1, mode='train')
    #trial_data = ImageDataset(normalize=True, flatten=False, max_size=224, dataselect=list(range(8000,9999)), mode='train')
    #trial_data = ImageDataset(normalize=True, flatten=False, max_size=224, dataselect=list(range(8000,10000)), mode='train')
    ans_from_model = predict(model, trial_data, predict_iteration=args.iteration, minibatch_size=1, device=args.gpu)
    output_submit_file(ans_from_model, output_filename, trial_data)
