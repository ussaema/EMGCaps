from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    """
        Transforms raw EMG/IMU samples into a feature vector used for classification
    """

    requires_global_setup   = False

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    @abstractmethod
    def extract_feature_point(self, raw_samples):
        """
        :param raw_samples: A window of emg samples.
        :return: A single feature point.
        """
        pass

    @abstractmethod
    def global_setup(self, all_raw_samples):
        """
              :param all_raw_samples: All windows of emg samples.
        """
        pass