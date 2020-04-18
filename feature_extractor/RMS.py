from . import FeatureExtractor
import numpy as np

class RMS(FeatureExtractor):

    def extract_feature_point(self, raw_samples):
        return np.sqrt(np.mean(np.square(raw_samples), axis=0))

    # Window based RMS
    #
    # def extract_feature_point(self, raw_samples):
    #     split_data      = np.split(raw_samples, indices_or_sections = 4, axis=0)
    #     mag_feat_list   = np.array([])
    #
    #     for data in split_data:
    #         mag_feat_list  = np.append(mag_feat_list, np.sqrt(np.mean(np.square(data), axis=0)))
    #
    #     return mag_feat_list

    def global_setup(self, all_raw_samples):
        pass