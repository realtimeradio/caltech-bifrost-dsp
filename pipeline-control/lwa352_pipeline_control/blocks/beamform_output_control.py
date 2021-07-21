from .block_control_base import BlockControl

class BeamformOutputControl(BlockControl):
    def set_destination(self, dest_ips, dest_ports):
        assert isinstance(dest_ips, list)
        assert isinstance(dest_ports, list)
        self._send_command(
            dest_ip = dest_ips,
            dest_port = dest_ports,
        )
