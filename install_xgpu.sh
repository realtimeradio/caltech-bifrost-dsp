#!/bin/bash

CURDIR=`pwd`
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPTDIR/xgpu/src; make clean && make NSTATION=352 NFREQUENCY=192 NTIME=480 NTIME_PIPE=480 CUDA_ARCH=sm_75 DP4A=yes DEVSWIZZLE=yes && sudo make install
cd $CURDIR
