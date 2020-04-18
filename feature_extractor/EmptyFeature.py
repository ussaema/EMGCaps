from . import FeatureExtractor


class EmptyFeature(FeatureExtractor):
    """
        Raw EMG window samples
    """

    def extract_feature_point(self, raw_samples):
        return raw_samples

    def global_setup(self, all_raw_samples):
        pass