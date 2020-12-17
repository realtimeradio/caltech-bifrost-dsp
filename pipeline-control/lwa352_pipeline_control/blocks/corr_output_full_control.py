from .block_control_base import BlockControl

class CorrOutputFull(BlockControl):
    def set_destination(self, dest_ip, dest_port):
        """
        Set the destination IP and UDP port for correlator
        packets, using the ``dest_ip`` and ``dest_port`` keys.

        :param dest_ip: Desired destination IP address, in dotted
            quad notation -- eg. "10.10.0.1"
        :type dest_ip: str

        :param dest_port: Desired destination UDP port
        :type dest_port: int

        """

        assert isinstance(dest_ip, str)
        assert isinstance(dest_port, int)
        self._send_command(
            dest_ip = dest_ip,
            dest_port = dest_port,
        )

    def set_max_mbps(self, max_mbps):
       """
       Use the ``max_mbps`` key to throttle the correlator
       output to at most ``max_mbps`` megabits per second.

       :param max_mbps: Output data rate cap, in megabits per second
       :type max_mbps: int
       """

       assert isinstance(max_mbps, int)
       self._send_command(max_mbps=max_mbps)
