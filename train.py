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
from predict_dataset import *
from net.nets import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                            help='GPU ID (negative value indicates CPU)')
    
    parser.add_argument('--net', '-n', default='CNN', help='Choose network to train')
    parser.add_argument('--epoch', '-e', default=200, help='Numbers of learning epoch')
    parser.add_argument('--data_dir', '-i', default='data', help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--model_snapshot_interval', '-m', type=int, default=0,
                                help='Interval of model snapshot')
    parser.add_argument('--trainer_snapshot_interval', '-t', type=int, default=0,
                                help='Interval of trainer snapshot')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    args = parser.parse_args()
    
    train_data = ImageDataset(normalize=True, flatten=False, max_size=224, dataselect=-1, data_dir=args.data_dir)
    test_data = ImageDataset(normalize=True, flatten=False, max_size=224, dataselect=-1, data_dir=args.data_dir)
    
    train_iter = iterators.SerialIterator(train_data, batch_size=50, repeat=True, shuffle=True)
    test_iter = iterators.SerialIterator(test_data, batch_size=50, repeat=False, shuffle=True)

    predictor = ''
    model = ''
    if args.net == 'CNN':
        predictor = CNN()
        model = L.Classifier(predictor)
    elif args.net == 'GoogLeNet':
        model = GoogLeNetBN()
    else:
        print('Such network is not defined')
        exit()
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
    
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (int(args.epoch), 'epoch'), out=args.out)
    
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), log_name='log.txt'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    
    if args.model_snapshot_interval > 0:
        if args.net == 'CNN':
            trainer.extend(extensions.snapshot_object(predictor, args.net + '_epoch_{.updater.epoch}.npz'), trigger=(args.model_snapshot_interval, 'epoch'))
        else:
            trainer.extend(extensions.snapshot_object(model, args.net + '_epoch_{.updater.epoch}.npz'), trigger=(args.model_snapshot_interval, 'epoch'))
             
    if args.trainer_snapshot_interval > 0:
        trainer.extend(extensions.snapshot(filename = 'trainer_epoch_{.updater.epoch}.npz'), trigger=(args.trainer_snapshot_interval, 'epoch'))
    
    if args.resume:
        # Resume from a snapshot
        print("resume from " + args.resume)
        chainer.serializers.load_npz(args.resume, trainer)
    
    print("start running")
    trainer.run()
    print("end running")
