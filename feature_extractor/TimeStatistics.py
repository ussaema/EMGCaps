from . import FeatureExtractor
import numpy as np

class TimeStatistics(FeatureExtractor):
    """
        Refer to:
            'Hudgins B, Parker P, Scott RN. A new strategy for multifunction myoelectric control. Biomedical
                Engineering, IEEE Transactions on. 1993; 40(1):82–94. https://doi.org/10.1109/10.204774'

        -> Following time domain statistics are computed
            1. Mean absolute value
            2. Number of zeroes (with a threshold)
            3. Number of slope changes (with a threshold)
            4. Waveform length
    """
    noise_thresh = 2

    # Note: We do not use "Mean Absolute Value Slope"
    def extract_feature_point(self, raw_samples):

        window_size     = raw_samples.shape[0]
        num_channels    = raw_samples.shape[1]

        mean_abs            = np.mean(np.abs(raw_samples), axis=0).astype(np.uint16)
        num_zeros           = np.zeros(num_channels, dtype=np.uint16)
        num_slope_changes   = np.zeros(num_channels, dtype=np.uint16)
        waveform_length     = np.zeros(num_channels, dtype=np.uint16)

        for i in range(num_channels):
            for j in range(window_size):

                # Check for zero crossings
                if (raw_samples[j][i] < self.noise_thresh) and (raw_samples[j][i] > -self.noise_thresh):
                    num_zeros[i] += 1

                # Check for slope changes
                if (j > 0) and (j < window_size - 1):
                    left    = raw_samples[j-1][i]
                    mid     = raw_samples[j][i]
                    right   = raw_samples[j+1][i]

                    condition_1 = (mid > left + self.noise_thresh) and (mid > right + self.noise_thresh)
                    condition_2 = (mid + self.noise_thresh < left) and (mid + self.noise_thresh < right)

                    if condition_1 or condition_2:
                        num_slope_changes[i] += 1

                # Compute waveform length
                if j > 0:
                    left    = raw_samples[j - 1][i]
                    mid     = raw_samples[j][i]
                    waveform_length[i] += np.abs(mid - left)

        # Concat time statistics features
        time_stat_vec = np.concatenate((mean_abs, num_zeros, num_slope_changes, waveform_length))
        return time_stat_vec

    def global_setup(self, all_raw_samples):
        pass
