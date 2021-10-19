#!/bin/bash

# Nvidia stuff needs to be done after each reboot.
# will be run once by nvidia_tweaks.service

LOG=/var/log/nvidia_tweaks.log

echo "`date --iso-8601=seconds`: 'Sleeping 120 sec before execution'" >> $LOG
sleep 120

echo "setupGPUhosts: Setting GPU power cap to 150W"
# Cap Turn on persistence mode
echo "`date --iso-8601=seconds`: 'Cap Turn on persistence mode'" >> $LOG
nvidia-smi -pm 1

# Cap GPU power to 150W
echo "`date --iso-8601=seconds`: 'Cap Cap GPU power to 150W'" >> $LOG
nvidia-smi -pl 150

# Set cpufreq to performance
echo "`date --iso-8601=seconds`: 'Set cpufreq to performance'" >> $LOG
for i in `ls /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
do
    echo "performance" > $i
done
