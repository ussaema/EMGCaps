from . import FeatureExtractor
import numpy as np
from sklearn.decomposition import PCA

class PaddedMultiRMS(FeatureExtractor):

    num_descriptors     = 5
    window_size         = 150
    num_channels        = 16
    pca_dim             = 16

    requires_global_setup   = True

    ####################################################################################################################
    ####################################################################################################################
    #
    # Non-PCA, non-overlapping window version
    #
    ####################################################################################################################
    ####################################################################################################################
    # def extract_feature_point(self, raw_samples):
    #     # Define data windows
    #     if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #         start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
    #     else:
    #         num_start_indices   = raw_samples.shape[0] // self.window_size
    #         start_indices       = [x * self.window_size for x in range(num_start_indices)]
    #     emg_data            = [raw_samples[y: y+self.window_size] for y in start_indices]
    #
    #     # Create RMS descriptors
    #     descriptors         = [np.sqrt(np.mean(np.square(data), axis=0)) for data in emg_data]
    #
    #     # Pad with zros
    #     num_miss            = self.num_descriptors - len(descriptors)
    #     descriptors        += [np.zeros(self.num_channels) for i in range(num_miss)]
    #     descriptors         = (np.array(descriptors)).flatten()
    #
    #     return descriptors
    #
    # def global_setup(self, all_raw_samples):
    #     pass
    ####################################################################################################################
    ####################################################################################################################


    ####################################################################################################################
    ####################################################################################################################
    #
    # PCA, overlapping window version
    #
    ####################################################################################################################
    ####################################################################################################################
    def extract_feature_point(self, raw_samples):
        # Define data windows
        start_indices = np.linspace(0, raw_samples.shape[0] - self.window_size, num=self.num_descriptors,
                                    endpoint=False,
                                    dtype=int)
        emg_data            = [raw_samples[y: y+self.window_size] for y in start_indices]

        # Create RMS descriptors
        descriptors = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
        norms       = np.linalg.norm(descriptors, axis=1)
        descriptors /= np.expand_dims(norms, axis=1)
        descriptors = self.pca.transform(descriptors)

        # Pad with zros
        num_miss        = self.num_descriptors - descriptors.shape[0]
        descriptors     = np.concatenate((descriptors, np.zeros((num_miss, self.pca_dim))), axis=0)
        descriptors     = descriptors.flatten()
        return descriptors

    def global_setup(self, all_raw_samples):

        all_descriptors = []

        for raw_samples in all_raw_samples:
            # Define data windows
            start_indices = np.linspace(0, raw_samples.shape[0] - self.window_size -1 , num=self.num_descriptors, endpoint=False,
                                        dtype=int)
            emg_data = [raw_samples[y: y + self.window_size] for y in start_indices]

            # Create RMS descriptors
            temp = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
            norm = np.linalg.norm(temp, axis=1)
            temp = temp / np.expand_dims(norm, axis=1)

            for i in range(temp.shape[0]):
                all_descriptors.append(temp[i])

        all_descriptors = np.array(all_descriptors)

        self.pca = PCA(n_components=self.pca_dim)
        self.pca.fit(all_descriptors)


    ####################################################################################################################
    ####################################################################################################################
    #
    # PCA, non-overlapping window version (ORIGINAL)
    #
    ####################################################################################################################
    ####################################################################################################################
    # def extract_feature_point(self, raw_samples):
    #     # Define data windows
    #     if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #         start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
    #     else:
    #         num_start_indices   = raw_samples.shape[0] // self.window_size
    #         start_indices       = [x * self.window_size for x in range(num_start_indices)]
    #     emg_data            = [raw_samples[y: y+self.window_size] for y in start_indices]
    #
    #     # Create RMS descriptors
    #     descriptors = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #     norms       = np.linalg.norm(descriptors, axis=1)
    #     descriptors /= np.expand_dims(norms, axis=1)
    #     descriptors = self.pca.transform(descriptors)
    #
    #     # Pad with zros
    #     num_miss        = self.num_descriptors - descriptors.shape[0]
    #     descriptors     = np.concatenate((descriptors, np.zeros((num_miss, self.pca_dim))), axis=0)
    #     descriptors     = descriptors.flatten()
    #     return descriptors
    #
    # def global_setup(self, all_raw_samples):
    #
    #     all_descriptors     = []
    #
    #     for raw_samples in all_raw_samples:
    #         # Define data windows
    #         if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #             start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False,
    #                                         dtype=int)
    #         else:
    #             num_start_indices = raw_samples.shape[0] // self.window_size
    #             start_indices = [x * self.window_size for x in range(num_start_indices)]
    #
    #         emg_data    = [raw_samples[y: y + self.window_size] for y in start_indices]
    #
    #         temp = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #         norm = np.linalg.norm(temp, axis=1)
    #         temp = temp / np.expand_dims(norm, axis=1)
    #         for i in range(temp.shape[0]):
    #             all_descriptors.append(temp[i])
    #
    #     all_descriptors = np.array(all_descriptors)
    #     self.pca = PCA(n_components=self.pca_dim)
    #     self.pca.fit(all_descriptors)
    ####################################################################################################################
    ####################################################################################################################


    ####################################################################################################################
    ####################################################################################################################
    #
    # Non-overlapping windows, PCA performed on zero padded vectors
    #
    ####################################################################################################################
    ####################################################################################################################
    # def extract_feature_point(self, raw_samples):
    #     # Define data windows
    #     if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #         start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
    #     else:
    #         num_start_indices = raw_samples.shape[0] // self.window_size
    #         start_indices = [x * self.window_size for x in range(num_start_indices)]
    #     emg_data = [raw_samples[y: y + self.window_size] for y in start_indices]
    #
    #     # Create RMS descriptors
    #     descriptors = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #     norms = np.linalg.norm(descriptors, axis=1)
    #     descriptors /= np.expand_dims(norms, axis=1)
    #
    #     # Pad with zros
    #     num_miss = self.num_descriptors - descriptors.shape[0]
    #     descriptors = np.concatenate((descriptors, np.zeros((num_miss, self.num_channels))), axis=0)
    #     descriptors = descriptors.flatten()
    #     descriptors = np.squeeze(self.pca.transform(descriptors.reshape(1, -1)))
    #
    #     return descriptors
    #
    # def global_setup(self, all_raw_samples):
    #
    #     all_descriptors = []
    #
    #     for raw_samples in all_raw_samples:
    #         # Define data windows
    #         if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
    #             start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False,
    #                                         dtype=int)
    #         else:
    #             num_start_indices = raw_samples.shape[0] // self.window_size
    #             start_indices = [x * self.window_size for x in range(num_start_indices)]
    #         emg_data = [raw_samples[y: y + self.window_size] for y in start_indices]
    #
    #         # Create RMS descriptors
    #         temp = np.array([np.mean(np.abs(data), axis=0) for data in emg_data])
    #         norm = np.linalg.norm(temp, axis=1)
    #         temp = temp / np.expand_dims(norm, axis=1)
    #
    #         num_miss = self.num_descriptors - temp.shape[0]
    #         descriptors = np.concatenate((temp, np.zeros((num_miss, self.num_channels))), axis=0)
    #         all_descriptors.append(descriptors.flatten())
    #
    #     all_descriptors = np.array(all_descriptors)
    #
    #     self.pca = PCA(n_components=self.pca_dim)
    #     self.pca.fit(all_descriptors)
    ####################################################################################################################
    ####################################################################################################################


