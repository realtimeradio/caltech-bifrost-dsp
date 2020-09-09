from blocks.block_control_base import BlockControl

class CorrOutputFull(BlockControl):
    def set_destination(self, dest_ip, dest_port):
        assert isinstance(dest_ip, str)
        assert isinstance(dest_port, int)
        self.send_command(
            dest_ip = dest_ip,
            dest_port = dest_port,
        )
