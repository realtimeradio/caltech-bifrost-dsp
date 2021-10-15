#!/bin/bash

# Nvidia stuff needs to be done after each reboot.
# will be run once by nvidia_tweaks.service

echo "setupGPUhosts: Setting GPU power cap to 150W"
#Cap Turn on persistence mode
nvidia-smi -pm 1

#Cap GPU power to 150W
nvidia-smi -pl 150
