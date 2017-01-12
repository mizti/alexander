import numpy as np
import argparse
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from image_dataset import *

class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(in_channels=3, out_channels=16, ksize=5, stride=2, pad=0),
            conv2 = L.Convolution2D(in_channels=16, out_channels=32, ksize=3, stride=1, pad=0),
            conv3 = L.Convolution2D(in_channels=32, out_channels=64, ksize=3, stride=1, pad=0),
            l1 = L.Linear(7744, 512),
            l2 = L.Linear(512, 25)
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.l1(h))
        y = self.l2(h)
        return y

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

train_data = ImageDataset(normalize=True, flatten=False, train=True, max_size=200)
test_data = ImageDataset(normalize=True, flatten=False, train=False, max_size=200)

train_iter = iterators.SerialIterator(train_data, batch_size=201, shuffle=True)
test_iter = iterators.SerialIterator(test_data, batch_size=201, repeat=False, shuffle=False)

model = L.Classifier(CNN())

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU

optimizer = optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

trainer = training.Trainer(updater, (500, 'epoch'), out='result')
print("start running")
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
#trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
print("end running")
