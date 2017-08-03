# Thank you to this lovely blog post for some hdf5 data gen guidance:
# https://ceciliavision.wordpress.com/2016/03/21/caffe-hdf5-layer/

from __future__ import print_function
import h5py
import os
import scipy.misc
import skimage

print("generating hdf5 data")

DIR = os.getcwd()
print("this is DIR", DIR)

dataset_name = 'test'

h5_fn = os.path.join(DIR, dataset_name+'.h5')

X = scipy.misc.imread('00177.png')
Y = scipy.misc.imread('00177_graybel.png')

with h5py.File(h5_fn, 'w') as f:
   f['data'] = skimage.img_as_float(X)
   f['label'] = skimage.img_as_float(Y)

text_fn = os.path.join(DIR, dataset_name+'.txt')
with open(text_fn, 'w') as f:
   print(h5_fn, file = f)
