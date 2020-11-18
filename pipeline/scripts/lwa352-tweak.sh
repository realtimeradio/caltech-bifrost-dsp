sudo ifconfig ens1f1 100.100.100.101/24 mtu 9000
sudo sysctl -w net.core.rmem_max=26214400
sudo sysctl -w net.core.rmem_default=26214400
