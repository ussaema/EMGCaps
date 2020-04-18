from . import FeatureExtractor
import numpy as np
import scipy.cluster.vq as sci_c

class KMeansRMS(FeatureExtractor):
    """
        A variation of NetVLAD, where clusters are computed, and the distance to the clusters are used as a
            similarity measure (instead of some complex function).
    """

    num_channels    = 16
    num_clusters    = 80
    window_size     = 200
    num_descriptors = 5

    requires_global_setup   = True

    def extract_feature_point(self, raw_samples):

        # Define data windows
        if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
            start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
        else:
            num_start_indices = raw_samples.shape[0] // self.window_size
            start_indices = [x * self.window_size for x in range(num_start_indices)]
        emg_data = [raw_samples[y: y + self.window_size] for y in start_indices]

        # Create RMS descriptors
        descriptors = [np.sqrt(np.mean(np.square(data), axis=0)) for data in emg_data]
        descriptors = np.array(descriptors)
        descriptors = np.expand_dims(descriptors, axis=0)
        clust_dist  = descriptors - self.clusters
        dist_norm   = np.linalg.norm(clust_dist, axis=2)
        max_norm    = np.max(dist_norm, axis=0)
        dist_norm  /= max_norm
        dist_norm   = np.sum(dist_norm, axis=1)
        dist_norm  /= np.linalg.norm(dist_norm)
        return dist_norm

    def global_setup(self, all_raw_samples):

        all_descriptors = []

        for raw_samples in all_raw_samples:
            # Define data windows
            if raw_samples.shape[0] // self.window_size >= self.num_descriptors:
                start_indices = np.linspace(0, raw_samples.shape[0], num=self.num_descriptors, endpoint=False, dtype=int)
            else:
                num_start_indices = raw_samples.shape[0] // self.window_size
                start_indices = [x * self.window_size for x in range(num_start_indices)]
            emg_data = [raw_samples[y: y + self.window_size] for y in start_indices]

            # Create RMS descriptors
            all_descriptors.extend([np.sqrt(np.mean(np.square(data), axis=0)) for data in emg_data])

        all_descriptors = np.array(all_descriptors)
        self.clusters   = sci_c.kmeans(all_descriptors, k_or_guess=self.num_clusters, iter=100)[0]
        self.clusters   = np.expand_dims(self.clusters, axis=1)