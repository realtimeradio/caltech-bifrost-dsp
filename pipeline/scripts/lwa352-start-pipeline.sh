NCHAN=96
IFACE=(enp216s0 enp216s0 enp24s0 enp24s0)
RXPORT=(10000 20000 10000 20000)
GPU=(0 0 1 1)
BUFGBYTES=4
ETCDHOST=etcdv3service.sas.pvt
CORES=("1,2,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4" "6,7,8,9,9,9,9,9,9,9,9,9,9,9,9" "11,12,13,14,14,14,14,14,14,14,14,14,14,14" "16,17,18,19,19,19,19,19,19,19,19,19,19,19,19")
CPUMASK=(0x3fe 0x3fe 0xff800 0xff800)

get_ip () {
  echo `ip addr show $1 | grep -o "inet [0-9]*\.[0-9]*\.[0-9]*\.[0-9]*" | grep -o "[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*"`
}

make_cmd () {
  IP=`get_ip ${IFACE[$1]}`
  LOGFILE=~/`hostname`.$1.log
  COMMAND=" \
    taskset ${CPUMASK[$1]} \
    lwa352-pipeline.py \
    --nchan $NCHAN \
    --ibverbs \
    --gpu ${GPU[$1]} \
    --pipelineid $1 \
    --useetcd \
    --etcdhost $ETCDHOST \
    --ip $IP \
    --port ${RXPORT[$1]} \
    --bufgbytes $BUFGBYTES \
    --cores ${CORES[$1]}
    --logfile $LOGFILE \
  "
  echo $COMMAND
}

for v in "$@"
do
  cmd=`make_cmd $v`
  echo $cmd
  $cmd &
done
