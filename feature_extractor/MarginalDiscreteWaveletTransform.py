from . import FeatureExtractor
import numpy as np
from pywt import wavedec

class MarginalDiscreteWaveletTransform(FeatureExtractor):
    """
        Refer to:
            'Lucas M, Gaufriau A, Pascual S, Doncarli C, Farina D. Multi-channel surface EMG classification using
                support vector machines and signal-based wavelet optimization. Biomedical Signal Processing and
                Control. 2008; 3(2):169–174. https://doi.org/10.1016/j.bspc.2007.09.002'

        Compute a discrete wavelet transform, and sums over each level of the transform (rectified)
    """

    num_levels      = 3
    mother_wavelet  = "db7"

    def extract_feature_point(self, raw_samples):

        num_channels    = raw_samples.shape[1]
        all_coeff       = []

        for i in range(num_channels):
            coeffs = wavedec(raw_samples[:, i], self.mother_wavelet, level=self.num_levels)

            # "Marginal" of each level
            for j in range(self.num_levels):
                all_coeff.append(np.sum(np.abs(coeffs[j])))

        all_coeff   = np.array(all_coeff)
        return all_coeff

    def global_setup(self, all_raw_samples):
        pass
