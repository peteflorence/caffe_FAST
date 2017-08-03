# Fast-Net Up-Convolutional Network 

Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```
  make -j8
  make py
  make test -j8
  make runtest -j8
  ```

 Our Code includes the PR2016 so this error is expected during compilation  

 syncedmem.cpp:16] Check failed: error == cudaSuccess (29 vs. 0) driver shutting down


For training we use hdf5 files where we subtract the mean BGR values. 

For Training follow the steps: 

1) Create hdf5 files

2) Change prototxt files in the folder caffe_FAST/models/

3) Change path to caffe root at the solve files

4) Run Train_ALL.bash for train all refinements


For Testing follow the steps:

1) at models/eval.py change caffe_root

2) Change to appropriate deploy file and caffe models

3) python eval.py

4) Image and segmentation mask will be displayed and Seg_Out.jpg will be created