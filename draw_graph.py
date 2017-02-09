# coding: utf-8
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json

file_names = [
    'result/log.txt'
]

data_names = [
    'main/accuracy',
    'validation/main/accuracy',
    #'main/loss',
    "main/loss",
    "validation/main/loss",
]

data = {}

for filename in file_names:
    f = open(filename)
    #data[name] = json.load(f)
    js = json.load(f)
    for index, dataname in enumerate(data_names):
        #data[filename + '/' + dataname] = list(map(lambda x:x[dataname], js))
        data[dataname] = list(map(lambda x:x[dataname], js))

    f.close()

#markers = {
#    'result/log.txt': "1"
#}

for name in data:
    plt.plot(data[name], markevery=100, label=name)

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(loc="lower right")
plt.savefig('result/graph.png', bbox_inches='tight')
