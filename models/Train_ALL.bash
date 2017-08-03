#!/bin/bash 


#train FCN32s

python solve2.py 2>&1| tee FCN32sLOG_KITTY_1024_3vs3.log

cp fcn32_1024_3vs3/FCN_32S_snapshot_iter_300000.caffemodel .


#train FCN16s

python solve3.py 2>&1| tee FCN16sLOG_KITTY_1024_3vs3.log

cp fcn16_1024_3vs3/FCN_16S_snapshot_iter_300000.caffemodel .


#train FCN8s

python solve4.py 2>&1| tee FCN8sLOG_KITTY_1024_3vs3.log

cp fcn8_1024_3vs3/FCN_8S_snapshot_iter_300000.caffemodel .


#train FCN4s

python solve5.py 2>&1| tee FCN4sLOG_KITTY_1024_3vs3.log

cp fcn4_1024_3vs3/FCN_4S_snapshot_iter_300000.caffemodel .

#train FCN2s


python solve6.py 2>&1| tee FCN2sLOG_KITTY_1024_3vs3.log

cp fcn2_1024_3vs3/FCN_2S_snapshot_iter_300000.caffemodel .


#train FCN2s - RELU


python solve7.py 2>&1| tee FCN2sLOG_KITTY_1024_3vs3.log

cp fcn2_1024_3vs3_RELU_DROPOUT/FCN_2S_RELU_snapshot_iter_300000.caffemodel .


#train FCN2s - TWO outputs


python solve8.py 2>&1| tee FCN2sLOG_KITTY_1024_3vs3.log

cp fcn2_1024_TWO_OUTPUTS/FCN_2S_TWO_snapshot_iter_300000.caffemodel .
