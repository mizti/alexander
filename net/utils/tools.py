import os
import six

def download_model(path, model_name):
    if model_name == 'alexnet':
        url = 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'
        name = 'bvlc_alexnet.caffemodel'
    elif model_name == 'caffenet':
        url = 'http://dl.caffe.berkeleyvision.org/' \
              'bvlc_reference_caffenet.caffemodel'
        name = 'bvlc_reference_caffenet.caffemodel'
    elif model_name == 'googlenet':
        url = 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
        name = 'bvlc_googlenet.caffemodel'
    elif model_name == 'resnet':
        #url = 'http://research.microsoft.com/en-us/um/people/kahe/resnet/models.zip'
        url = 'https://s3-ap-northeast-1.amazonaws.com/alexandermodeldata/ResNet-152-model.caffemodel'
        name = 'ResNet-152-model.caffemodel'
    else:
        raise RuntimeError('Invalid model type. Choose from '
                           'alexnet, caffenet, googlenet and resnet.')

    if os.path.isfile(path + '/' + name):
        print('passed!')
        pass
    else:
        print('Downloading model file...')
        six.moves.urllib.request.urlretrieve(url, path + '/' + name)
        print('Download completed')
        #if model_name == 'resnet':
        #    print('extracting file..')
        #    with zipfile.ZipFile(path + '/' + name, 'r') as zf:
        #        zf.extractall('.')
        print('Done.')
    return path + '/' + name
