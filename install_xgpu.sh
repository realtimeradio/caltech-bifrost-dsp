#!/bin/bash

CURDIR=`pwd`
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPTDIR/xgpu/src; make clean && make NSTATION=2048 NFREQUENCY=32 NTIME=64 NTIME_PIPE=64 CUDA_ARCH=sm_75 DP4A=yes DEVSWIZZLE=yes && make install
cd $CURDIR
