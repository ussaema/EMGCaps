from . import FeatureExtractor
import numpy as np

class FourierBinning(FeatureExtractor):
    """
        Compute discrete Fourier transform, compute magnitude of returned Fourier coefficients, and compute sums
            of groups of coefficients.
    """

    requires_global_setup   = False
    custom_mask             = []

    # def __init__(self):
    #
    #     #
    #     # base_mask = [0, 1, ..., 10, 11 (appears 10 times), 12 (appears 10 times), ..., 19 (appears 10 times)]
    #     #
    #     final_mask  = []
    #     base_mask   = [0]
    #     base_mask.extend([1 for x in range(5)])
    #     base_mask.extend([2 for x in range(5)])
    #     base_mask.extend([3 for x in range(5)])
    #     base_mask.extend([4 for x in range(10)])
    #
    #     for j in range(16):
    #         if j > 0:
    #             for k in range(len(base_mask)):
    #                 base_mask[k] += 5
    #         final_mask.extend(base_mask)
    #     self.custom_mask = final_mask
    #
    # def extract_feature_point(self, raw_samples):
    #     split_data      = np.split(raw_samples, indices_or_sections = 4, axis=0)
    #     mag_feat_list   = np.array([])
    #
    #     for data in split_data:
    #         ft_feat     = np.fft.rfft(data, axis=0, norm="ortho")
    #         mag_feat    = np.abs(ft_feat)
    #         mag_feat    = mag_feat.flatten("F")
    #         mag_feat    = np.bincount(self.custom_mask, weights=mag_feat)
    #         mag_feat_list = np.append(mag_feat_list, mag_feat)
    #
    #     return mag_feat_list
    #
    # def global_setup(self, all_raw_samples):
    #     # shape = all_raw_samples.shape
    #     # sample  = np.reshape(all_raw_samples, (shape[0], shape[1]*shape[2]))
    #     # #sample  = np.reshape(sample, (40*16))
    #     # pca = PCA()
    #     # z   = pca.fit(sample)
    #     pass


    def __init__(self, dataset_name):
        #
        # base_mask = [0, 1, ..., 10, 11 (appears 10 times), 12 (appears 10 times), ..., 19 (appears 10 times)]
        #
        super().__init__(dataset_name)
        final_mask  = []
        base_mask   = [0]
        base_mask.extend([1 for x in range(10)])
        base_mask.extend([2 for x in range(10)])
        base_mask.extend([3 for x in range(20)])
        base_mask.extend([4 for x in range(20)])
        base_mask.extend([5 for x in range(40)])

        for j in range(16):
            if j > 0:
                for k in range(len(base_mask)):
                    base_mask[k] += 6
            final_mask.extend(base_mask)

        self.custom_mask = final_mask

    def extract_feature_point(self, raw_samples):
        ft_feat     = np.fft.rfft(raw_samples, axis=0, norm="ortho")
        mag_feat    = np.abs(ft_feat)
        mag_feat    = mag_feat.flatten("F")
        mag_feat    = np.bincount(self.custom_mask, weights=np.array(mag_feat))
        return mag_feat

    def global_setup(self, all_raw_samples):
        pass