from .block_control_base import BlockControl

class BeamformVlbiOutputControl(BlockControl):
    def set_destination(self, dest_ip, dest_port):
        assert isinstance(dest_ip, str)
        assert isinstance(dest_port, int)
        return self._send_command(
            dest_ip = dest_ip,
            dest_port = dest_port,
        )

    #def set_packet_delay(self, delay_ns):
    #   assert _isinstance(delay_ns, int)
    #   self.send_command(packet_delay_ns=delay_ns)
