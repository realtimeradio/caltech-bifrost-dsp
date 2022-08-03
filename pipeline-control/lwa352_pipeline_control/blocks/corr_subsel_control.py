from .block_control_base import BlockControl
import simplejson as json
import numpy as np

class CorrSubselControl(BlockControl):
    nvis_out = 48 * 49 * 4 // 2 #48 stand dual pol
    def set_baseline_select(self, subsel):
        """
        Set which baselines should be output from the
        fast correlator pipeline.

        Parameters:
        subsel : A numpy array, or 3D list, of dimensions <n_visibility>,2,2

        First dimension runs over number of visibilities to be selected.
        Second dimension runs over input pair (0=unconjugated input. 1=conjugated input)
        Third dimension runs over stand/pol.

        Example:
        To set the baseline subsection to choose:

        - visibility 0: the autocorrelation of antenna 0, polarization 0
        - visibility 1: the cross correlation of antenna 5, polarization 1 with antenna 6, polarization 0

        use:

        ``subsel = [ [[0,0], [0,0]], [[5,1], [6,0]], ... ]``
        """
        subsel = np.array(subsel, dtype=np.int32)
        assert subsel.shape == (self.nvis_out, 2, 2)
        return self._send_command(baselines=subsel.tolist())

    def get_baseline_select(self):
        """
        Attempt to retrieve the currently loaded baseline selection array.
        This requires data to be flowing through the pipeline.
        """
        stats = self.get_bifrost_status()['stats']
        if not 'baselines' in stats:
            return None
        else:
            baselines = json.loads(stats['baselines'])
            return np.array(baselines, dtype=int)
