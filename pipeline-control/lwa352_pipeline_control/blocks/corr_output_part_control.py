from .block_control_base import BlockControl

class CorrOutputPartControl(BlockControl):
    def set_destination(self, dest_ip, dest_port):
        assert isinstance(dest_ip, str)
        assert isinstance(dest_port, int)
        self._send_command(
            dest_ip = dest_ip,
            dest_port = dest_port,
        )

    def set_packet_delay(self, delay_ns):
       assert isinstance(delay_ns, int)
       self._send_command(packet_delay_ns=delay_ns)
