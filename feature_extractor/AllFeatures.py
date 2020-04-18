from . import FeatureExtractor
from .TimeStatistics import TimeStatistics
from .RMS import RMS
from .HistogramBins import HistogramBins
from .MarginalDiscreteWaveletTransform import MarginalDiscreteWaveletTransform
import numpy as np

class AllFeatures(FeatureExtractor):
    """
        A simple concatenation of all baseline features
    """

    requires_global_setup = True

    def __init__(self):
        self.ts     = TimeStatistics()
        self.rms    = RMS()
        self.hist   = HistogramBins()
        self.mdwt   = MarginalDiscreteWaveletTransform()

    def extract_feature_point(self, raw_samples):

        ts_feat     = self.ts.extract_feature_point(raw_samples)
        rms_feat    = self.rms.extract_feature_point(raw_samples)
        hist_feat   = self.hist.extract_feature_point(raw_samples)
        mdwt_feat   = self.mdwt.extract_feature_point(raw_samples)

        return np.concatenate((ts_feat, rms_feat, hist_feat, mdwt_feat))

    def global_setup(self, all_raw_samples):
        self.hist.global_setup(all_raw_samples)