from . import FeatureExtractor
import numpy as np

class MultipleRMS(FeatureExtractor):

    num_descriptors     = 4
    window_size         = 150
    num_channels        = 16

    def extract_feature_point(self, raw_samples):
        start_indices   = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
        emg_data        = [raw_samples[y:y+self.window_size] for y in start_indices]
        descriptors     = [np.sqrt(np.mean(np.square(data), axis=0)) for data in emg_data]
        descriptors     = np.array(descriptors)

        return descriptors.flatten()

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
