import numpy as np
import argparse
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from datetime import datetime

class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(in_channels=3, out_channels=16, ksize=5, stride=2, pad=0),
            norm1 = L.BatchNormalization(16),
            conv2 = L.Convolution2D(in_channels=16, out_channels=32, ksize=3, stride=1, pad=0),
            norm2 = L.BatchNormalization(32),
            conv3 = L.Convolution2D(in_channels=32, out_channels=64, ksize=3, stride=1, pad=0),
            norm3 = L.BatchNormalization(64),
            l1 = L.Linear(18496, 1200),
            l2 = L.Linear(1200, 25)
        )

    def __call__(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.norm2(self.conv2(h)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.norm3(self.conv3(h)))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.l1(h))
        y = self.l2(h)
        return y



class GoogLeNetBN(chainer.Chain):

    """New GoogLeNet of BatchNormalization version."""

    insize = 224

    def __init__(self):
        super(GoogLeNetBN, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, stride=2, pad=3, nobias=True),
            norm1=L.BatchNormalization(64),
            conv2=L.Convolution2D(64, 192, 3, pad=1, nobias=True),
            norm2=L.BatchNormalization(192),
            inc3a=L.InceptionBN(192, 64, 64, 64, 64, 96, 'avg', 32),
            inc3b=L.InceptionBN(256, 64, 64, 96, 64, 96, 'avg', 64),
            inc3c=L.InceptionBN(320, 0, 128, 160, 64, 96, 'max', stride=2),
            inc4a=L.InceptionBN(576, 224, 64, 96, 96, 128, 'avg', 128),
            inc4b=L.InceptionBN(576, 192, 96, 128, 96, 128, 'avg', 128),
            inc4c=L.InceptionBN(576, 128, 128, 160, 128, 160, 'avg', 128),
            inc4d=L.InceptionBN(576, 64, 128, 192, 160, 192, 'avg', 128),
            inc4e=L.InceptionBN(576, 0, 128, 192, 192, 256, 'max', stride=2),
            inc5a=L.InceptionBN(1024, 352, 192, 320, 160, 224, 'avg', 128),
            inc5b=L.InceptionBN(1024, 352, 192, 320, 192, 224, 'max', 128),
            out=L.Linear(1024, 25),

            conva=L.Convolution2D(576, 128, 1, nobias=True),
            norma=L.BatchNormalization(128),
            lina=L.Linear(2048, 1024, nobias=True),
            norma2=L.BatchNormalization(1024),
            outa=L.Linear(1024, 25),

            convb=L.Convolution2D(576, 128, 1, nobias=True),
            normb=L.BatchNormalization(128),
            linb=L.Linear(2048, 1024, nobias=True),
            normb2=L.BatchNormalization(1024),
            outb=L.Linear(1024, 25),
        )
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        self.inc3a.train = value
        self.inc3b.train = value
        self.inc3c.train = value
        self.inc4a.train = value
        self.inc4b.train = value
        self.inc4c.train = value
        self.inc4d.train = value
        self.inc4e.train = value
        self.inc5a.train = value
        self.inc5b.train = value

    def predict(self, x):
        test = not self._train

        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x), test=test)),  3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h), test=test)), 3, stride=2, pad=1)
        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)
        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.out(h)
        h = F.softmax(h)
        return h.data

    def __call__(self, x, t):
        test = not self._train

        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x), test=test)),  3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h), test=test)), 3, stride=2, pad=1)
        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = F.relu(self.norma(self.conva(a), test=test))
        a = F.relu(self.norma2(self.lina(a), test=test))
        a = self.outa(a)
        loss1 = F.softmax_cross_entropy(a, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = F.relu(self.normb(self.convb(b), test=test))
        b = F.relu(self.normb2(self.linb(b), test=test))
        b = self.outb(b)
        loss2 = F.softmax_cross_entropy(b, t)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.out(h)
        loss3 = F.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = F.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy,
            'timestamp': int(datetime.now().strftime('%Y%m%d%H%M%S'))
        }, self)
        return loss


from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter


class ClassifierForResNet(link.Chain):

    """A simple classifier model.

    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.

    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.

    """

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(ClassifierForResNet, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, *args):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.

        The all elements of ``args`` but last one are features and
        the last element corresponds to ground truth labels.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """

        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)['fc6']
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
