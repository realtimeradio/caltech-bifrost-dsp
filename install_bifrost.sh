#!/bin/bash

CURDIR=`pwd`
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPTDIR/bifrost
./configure --enable-xgpu
make clean && make && sudo make install
cd $CURDIR
