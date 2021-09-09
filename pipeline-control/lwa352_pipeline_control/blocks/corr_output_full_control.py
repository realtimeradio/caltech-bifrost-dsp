from .block_control_base import BlockControl

class CorrOutputFullControl(BlockControl):
    """
    Control interface for the ``CorrOutputFull`` processing block.
    Control keys:

    .. table::
        :widths: 20 10 10 60

        +------------------+--------+---------+------------------------------+
        | Field            | Format | Units   | Description                  |
        +==================+========+=========+==============================+
        | ``dest_ip``      | string |         | Destination IP for           |
        |                  |        |         | transmitted packets, in      |
        |                  |        |         | dotted-quad format. Eg.      |
        |                  |        |         | ``"10.0.0.1"``. Use          |
        |                  |        |         | ``"0.0.0.0"`` to skip        |
        |                  |        |         | sending packets. See         |
        |                  |        |         | ``set_destination()``.       |
        +------------------+--------+---------+------------------------------+
        | ``dest_file``    | string |         | If not `""`, overrides       |
        |                  |        |         | ``dest_ip`` and causes the   |
        |                  |        |         | output data to be written to |
        |                  |        |         | the supplied file            |
        +------------------+--------+---------+------------------------------+
        | ``dest_port``    | int    |         | UDP port to which packets    |
        |                  |        |         | should be transmitted. See   |
        |                  |        |         | ``set_destination()``.       |
        +------------------+--------+---------+------------------------------+
        | ``max_mbps``     | int    | Mbits/s | The maximum output data rate |
        |                  |        |         | to allow before throttling.  |
        |                  |        |         | Set to ``-1`` to send as     |
        |                  |        |         | fast as possible. See        |
        |                  |        |         | ``set_max_mbps()``.          |
        +------------------+--------+---------+------------------------------+

    """
    def set_destination(self, dest_ip="0.0.0.0", dest_port=10000, dest_file=""):
        """
        Set the destination IP and UDP port for correlator
        packets, using the ``dest_ip`` and ``dest_port`` keys.

        :param dest_ip: Desired destination IP address, in dotted
            quad notation -- eg. "10.10.0.1"
        :type dest_ip: str

        :param dest_port: Desired destination UDP port
        :type dest_port: int

        :param dest_file: If provided, write data output packets to this
            file, rather than the destination IP. Useful for testing.
        :type dest_file: str

        """

        assert isinstance(dest_ip, str)
        assert isinstance(dest_port, int)
        assert isinstance(dest_file, str)
        return self._send_command(
            dest_ip = dest_ip,
            dest_port = dest_port,
            dest_file = dest_file,
        )

    def set_max_mbps(self, max_mbps):
       """
       Use the ``max_mbps`` key to throttle the correlator
       output to at most ``max_mbps`` megabits per second.
       Throttle is approximate only.

       :param max_mbps: Output data rate cap, in megabits per second
       :type max_mbps: int
       """

       assert isinstance(max_mbps, int)
       return self._send_command(max_mbps=max_mbps)

    def get_status(self):
        """
        Get correlator full visibility output stats:

        .. table::
            :widths: 25 10 10 55

            +----------------------+--------+---------+------------------------+
            | Field                | Type   | Unit    | Description            |
            +======================+========+=========+========================+
            | ``curr_sample``      | int    |         | The index of the last  |
            |                      |        |         | sample to be processed |
            +----------------------+--------+---------+------------------------+
            | ``dest_ip``          | string |         | Current destination IP |
            |                      |        |         | address, in dotted-    |
            |                      |        |         | quad notation.         |
            +----------------------+--------+---------+------------------------+
            | ``dest_port``        | int    |         | Current destination    |
            |                      |        |         | UDP port               |
            +----------------------+--------+---------+------------------------+
            | ``last_cmd_time``    | float  | UNIX    | The last time a        |
            |                      |        | time    | command was received   |
            +----------------------+--------+---------+------------------------+
            | ``last_update_time`` | float  | UNIX    | The last time settings |
            |                      |        | time    | from a command were    |
            |                      |        |         | loaded                 |
            +----------------------+--------+---------+------------------------+
            | ``max_mbps``         | int    | Mbits/s | The current throttle   |
            |                      |        |         | setpoint for output    |
            |                      |        |         | data                   |
            +----------------------+--------+---------+------------------------+
            | ``new_dest_ip``      | string |         | The commanded          |
            |                      |        |         | destination IP         |
            |                      |        |         | address, in dotted-    |
            |                      |        |         | quad notation. This IP |
            |                      |        |         | will be loaded on the  |
            |                      |        |         | next visibility matrix |
            |                      |        |         | to be transmitted if   |
            |                      |        |         | ``update_pending`` is  |
            |                      |        |         | True.                  |
            +----------------------+--------+---------+------------------------+
            | ``new_dest_port``    | int    |         | The commanded          |
            |                      |        |         | destination UDP port,  |
            |                      |        |         | to be loaded on the    |
            |                      |        |         | next visibility matrix |
            |                      |        |         | to be transmitted if   |
            |                      |        |         | ``update_pending`` is  |
            |                      |        |         | True                   |
            +----------------------+--------+---------+------------------------+
            | ``new_max_mbps``     | int    | Mbits/s | The commanded throttle |
            |                      |        |         | setpoint for output    |
            |                      |        |         | data, to be loaded on  |
            |                      |        |         | the next visibility    |
            |                      |        |         | matrix to be           |
            |                      |        |         | transmitted if         |
            |                      |        |         | ``update_pending`` is  |
            |                      |        |         | True                   |
            +----------------------+--------+---------+------------------------+
            | ``output_gbps``      | float  | Gbits/s | Measured output data   |
            |                      |        |         | rate for the last      |
            |                      |        |         | visibility matrix      |
            +----------------------+--------+---------+------------------------+
            | ``update_pending``   | bool   |         | Flag indicating that   |
            |                      |        |         | the IP/port/throttle   |
            |                      |        |         | settings have changed  |
            |                      |        |         | and should be updated  |
            |                      |        |         | on the next visibility |
            |                      |        |         | matrix                 |
            +----------------------+--------+---------+------------------------+

        :return: Block status dictionary
        :rtype: dict
        """

        return self._get_status()
