MAX_PIPELINES=2
IFACE=(enp24s0 enp216s0)
BUFGBYTES=4
ETCDHOST=etcdv3service.sas.pvt
CORES=("1,2,3,4,5,6,7,7,7,7,7,7,7" "11,12,13,14,15,16,17,17,17,17,17,17,17")
CPUMASK=(0x3fe 0xff800)

get_ip () {
  echo `ip addr show $1 | grep -o "inet [0-9]*\.[0-9]*\.[0-9]*\.[0-9]*" | grep -o "[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*"`
}

IP=`get_ip ${IFACE[$1]}`
LOGFILE=~/`hostname`_$1.log

COMMAND=\
"taskset ${CPUMASK[$1]} \
	lwa352-pipeline.py \
	--nobeamform \
	--ibverbs \
	--gpu $1 \
	--pipelineid $1 \
	--useetcd \
	--etcdhost $ETCDHOST \
	--ip $IP \
	--bufgbytes $BUFGBYTES \
	--cores ${CORES[$1]}
        --logfile $LOGFILE \
"

echo $COMMAND

$COMMAND
