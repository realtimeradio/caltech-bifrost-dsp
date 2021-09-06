#!/bin/bash

python ../../pipeline.py --fakesource --nobeamform --testdatain ../test_vectors/in_7200t_192c_352s_2p_deadbeef.dat -C 0,1,2,3,4,5,6,7,7,7,7,7 --testdatacorr ../test_vectors/corr_7200t_2400a_192c_352s_2p_deadbeef.dat --testdatacorr_acc_len 2400 -a 9600
