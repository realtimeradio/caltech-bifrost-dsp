#!/bin/bash

CURDIR=`pwd`
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPTDIR/xgpu/src; make clean && make NSTATION=2048 NFREQUENCY=16 NTIME=512 NTIME_PIPE=512 CUDA_ARCH=sm_86 DP4A=yes DEVSWIZZLE=yes && make install
cd $CURDIR
