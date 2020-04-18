from . import FeatureExtractor
from kymatio import Scattering1D
import torch
import numpy as np

class ScatteringFeat1D(FeatureExtractor):
    """
        Refer to (as well papers citing this paper):
            "J. Bruna and S. Mallat. Invariant scattering convolution networks.IEEE Transactions on Pattern Analysis
             and Machine Intelligence, 2013."

        Computes scattering transform features, which have useful mathematical properties for classification
    """

    # (1, T) -> (1, P, T/2**J)
    #   Where P ~= 1 + J Q + J (J-1) Q / 2.
    #
    J = 5       # Account for translation up to 2^6 samples
    T = 200     # Number of samples per feature vector
    Q = 2       # Resolution per octave

    def __init__(self):
        self.scattering_transform = Scattering1D(self.J, self.T, self.Q)

    def extract_feature_point(self, raw_samples):
        shape = raw_samples.shape
        raw_samples = np.reshape(raw_samples, (1, shape[1], shape[0]))
        raw_samples = torch.from_numpy(raw_samples).float()
        Sx = self.scattering_transform.forward(raw_samples)
        Sx = Sx.numpy()
        Sx = np.reshape(Sx, (Sx.shape[1] * Sx.shape[2] * Sx.shape[3]))
        return Sx

    def global_setup(self, all_raw_samples):
        pass