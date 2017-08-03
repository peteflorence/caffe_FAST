import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy

# Make sure that caffe is on the python path:
caffe_root = '/misc/lmbraid17/oliveira/caffe_Context_global/'  # this file is expected to be in 
import sys
sys.path.insert(0, caffe_root + 'python')

 
import caffe
 
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('000000_10.png')
im = im.resize((300,300),Image.ANTIALIAS)
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))
in_ = in_.reshape(1,3,300,300)


caffe.set_mode_gpu()
caffe.set_device(1)


# load net
#net = caffe.Net('fcn-4s-pascal-deploy_300.prototxt', 'FCN_4S_snapshot_iter_30000.caffemodel', caffe.TEST)
net = caffe.Net('fcn-4s-pascal-deploy_300.prototxt', 'FCN_4S_snapshot_iter_220000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, 3, 300,300)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['upscore'].data[0].argmax(axis=0)


#im = Image.fromarray(out)
#im.save("SEG_FCN16s.jpg")


scipy.misc.imsave('Seg_Out.jpg', out)

#Uncertainty
#for i in range(0, im.size[1]):
#	for j in range(0, im.size[0]):
#		if out[i,j] == 1:
#			new_color = im.getpixel( (j,i))
#			D=list(new_color)
#			D[1]=255
#			new_color=tuple(D)
#			im.putpixel( (j,i), new_color)


#scipy.misc.imsave('uu_000097_SEG2.jpg', im)

fig = plt.figure()
imgplot = plt.imshow(im)

fig = plt.figure()
plot = plt.imshow(out)
plt.show()







