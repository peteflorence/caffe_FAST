from __future__ import division
import numpy as np
#import matplotlib.pyplot as plt
#import Image

# Make sure that caffe is on the python path:
caffe_root = '/misc/lmbraid17/oliveira/caffe_FAST/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
 
# make a bilinear interpolation kernel
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
 
# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt
 
# base net -- follow the editing model parameters example to make
# a fully convolutional VGG16 net.
base_weights = 'FCN_32S_snapshot_iter_300000.caffemodel'
 
# init
caffe.set_mode_gpu()
caffe.set_device(0)
 
solver = caffe.SGDSolver('solverFCN16s_1024.prototxt')
 
# do net surgery to set the deconvolution weights for bilinear interpolation
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)
 
# copy base weights for fine-tuning
solver.net.copy_from(base_weights)
 
# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
solver.step(300000)
