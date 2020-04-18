import os
import numpy as np
from abc import ABC, abstractmethod
import copy
from tqdm import tqdm

#
# All datasets will use this interface, and implement "process_single_exercise()", "get_dataset_name()".
#
class Dataset(ABC):

    window_size     = 200
    overlap_size    = 100
    num_classes     = 52 + 1
    balance_classes = True # Will limit number of rest samples for train/test

    # Filled via "create_dataset()"
    train_features  = None
    train_labels    = None
    test_features   = None
    test_labels     = None
    all_samples     = None

    # Need to correct labels, by inspecting exercise number
    E1_classes      = 12
    E2_classes      = 17
    E3_classes      = 23
    E3_name         = "E3"
    E2_name         = "E2"
    rest_label      = 0

    # Data augmentation parameters
    snr_max     = 50 # Signal-to-Noise ratio = (variance of signal) / (variance of noise)
    snr_min     = 25
    snr_ticks   = 3 # Number of increments

    def __init__(self, all_data_path, feature_extractor, augment_data = True):

        if feature_extractor is None:
            raise ValueError("Feature extractor is empty.")
        if all_data_path is None:
            raise ValueError("All data path is empty.")

        self.feature_extractor  = feature_extractor
        self.all_data_path      = all_data_path
        self.augment_data       = augment_data

    def create_dataset(self, loaded_data, adjust_labels = True):

        if self.feature_extractor.requires_global_setup:
            self.create_dataset_helper(loaded_data, True, adjust_labels)

        self.create_dataset_helper(loaded_data, False, adjust_labels)

    def create_dataset_helper(self, loaded_data, obtain_all_samples, adjust_labels):
        """
            Converts loaded data (via NinaDataParser) into a useable, baseline dataset, consisting of:
                1. train_features
                2. train_labels
                3. test_features
                4. test_labels

        :param feature_extractor: A function that transform 40 emg samples (a window) into a single feature point.
        :param obtain_all_samples: Avoid creating a train/test split, simply obtain all samples of windowed data.
        :param adjust_labels: Adjust the ground truth labels (according to exercise number)
        """
        if self.load_dataset():
            return

        # To be filled
        self.train_features = []
        self.train_labels   = []
        self.test_features  = []
        self.test_labels    = []
        self.all_samples    = []

        # Class balancing
        num_samples         = 0
        num_rest_samples    = 0

        for patient in tqdm(loaded_data.keys()):
            for ex in loaded_data[patient].keys():
                self.process_single_exercise(loaded_data, patient, ex, num_samples,
                                                num_rest_samples, obtain_all_samples, adjust_labels)

        # Convert to numpy arrays:
        #
        if obtain_all_samples:
            #self.all_samples    = np.array(self.all_samples)
            self.feature_extractor.global_setup(self.all_samples)
        else:
            if self.augment_data:
                self.create_augmented_data(loaded_data, adjust_labels)

            self.train_features = np.array(self.train_features)#[:213]
            self.train_labels   = np.array(self.train_labels)#[:213]
            self.test_features  = np.array(self.test_features)#[:103]
            self.test_labels    = np.array(self.test_labels)#[:103]

            # Save the above to the baseline dataset directory:
            self.save_dataset()

    def save_dataset(self):

        feat_ext_name   = self.feature_extractor.__class__.__name__
        feat_path       = os.path.join(self.all_data_path, self.get_dataset_name(), feat_ext_name)

        if not os.path.exists(feat_path):
            os.makedirs(feat_path)

        if self.train_features is not None:
            np.save(os.path.join(feat_path, "train_features"), self.train_features)
        if self.train_labels is not None:
            np.save(os.path.join(feat_path, "train_labels"), self.train_labels)
        if self.test_features is not None:
            np.save(os.path.join(feat_path, "test_features"), self.test_features)
        if self.test_labels is not None:
            np.save(os.path.join(feat_path, "test_labels"), self.test_labels)

        if ((self.train_labels is None) or (self.train_labels is None) or
            (self.test_features is None) or (self.test_labels is None)):
            raise RuntimeError("One of the dataset pieces are empty.")

    def load_dataset(self):
        """
        :return: True, if this function was able to load the baseline dataset.
        """

        feat_ext_name   = self.feature_extractor.__class__.__name__
        feat_path       = os.path.join(self.all_data_path, self.get_dataset_name(), feat_ext_name)

        if not os.path.exists(feat_path):
            return False
        if not os.path.exists(os.path.join(feat_path, "train_features.npy")):
            return False
        if not os.path.exists(os.path.join(feat_path, "train_labels.npy")):
            return False
        if not os.path.exists(os.path.join(feat_path, "test_features.npy")):
            return False
        if not os.path.exists(os.path.join(feat_path, "test_labels.npy")):
            return False

        self.train_features = np.load(os.path.join(feat_path, "train_features.npy"))
        self.train_labels   = np.load(os.path.join(feat_path, "train_labels.npy"))
        self.test_features  = np.load(os.path.join(feat_path, "test_features.npy"))
        self.test_labels    = np.load(os.path.join(feat_path, "test_labels.npy"))

        return True

    def create_augmented_data(self, loaded_data, adjust_labels):
        print("Creating augmented data (slow)...")
        loaded_data_copy                = copy.deepcopy(loaded_data)
        channel_vars, num_increments    = self.get_channel_vars(loaded_data)

        # Create Gaussian white noise samples
        cov_mat         = np.diag(channel_vars)
        mean_vec        = np.zeros(16, np.float64)
        noise_samples   = np.random.multivariate_normal(mean_vec, cov_mat, num_increments)

        # Class balancing
        num_samples         = 0
        num_rest_samples    = 0

        # For each SNR combination
        for snr in tqdm(np.linspace(self.snr_min, self.snr_max, self.snr_ticks)):
            sample_counter  = 0
            np.random.shuffle(noise_samples)

            # Augment every single example
            for patient in loaded_data.keys():
                for ex in loaded_data[patient].keys():
                    # Augment exercise
                    sample_counter = self.augment_exercise(loaded_data_copy[patient][ex],
                                                            noise_samples, sample_counter, snr)

                    # Create noisy examples, add to self.train_features\labels...
                    self.process_single_exercise(loaded_data_copy, patient, ex, num_samples,
                                                 num_rest_samples, False, adjust_labels)

            loaded_data_copy = copy.deepcopy(loaded_data)

    def augment_exercise(self, exercise, noise_samples, sample_counter, snr):
        for idx, label in enumerate(exercise["restimulus"]):
            if label != 0:
                # The noise has variance y = x * (1/sqrt(SNR))^2 = x/SNR,
                #      where x is the variance of the signal, y is the variance of the noise.
                exercise["emg"][idx] += noise_samples[sample_counter] * (1 / np.sqrt(snr))
                sample_counter += 1
        return sample_counter

    def get_channel_vars(self, loaded_data):
        channel_vars    = np.zeros(16, dtype=np.float64)
        num_increments  = 0

        def increment_channel_sum(exercise, channel_sum):
            increments = 0
            for idx, label in enumerate(exercise["restimulus"]):
                if label != 0:
                    increments  += 1
                    channel_sum += np.square(exercise["emg"][idx]) # Assuming mean is 0
            return increments

        for patient in loaded_data.keys():
            for ex in loaded_data[patient].keys():
                num_increments += increment_channel_sum(loaded_data[patient][ex], channel_vars)

        return channel_vars / float(num_increments - 1), num_increments

    @abstractmethod
    def process_single_exercise(self, loaded_data, patient, ex, num_samples, num_rest_samples, obtain_all_samples,
                                    adjust_labels):
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass
