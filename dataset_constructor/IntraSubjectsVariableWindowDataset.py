from . import Dataset
import numpy as np
#
# The training/validation/testing dataset described in
#   "Comparison of six electromyography acquisition setups on hand movement classification tasks"
#       by Stefano Pizzolato, et al.
#
#   Note: Except here, the "window" is now the entire movement with corrected ground truth --> "Variable" Window
#
class IntraSubjectsVariableWindowDataset(Dataset):

    def process_single_exercise(self, loaded_data, patient, ex, num_samples, num_rest_samples, obtain_all_samples,
                                    adjust_labels):

        cur_data    = loaded_data[patient][ex]
        num_emg     = cur_data["emg"].shape[0]

        # Look for possible windows of EMG data
        #
        start_window = 0

        while start_window < num_emg:

            window_label = int(cur_data["restimulus"][start_window][0])

            offset = 0
            while (start_window + offset < num_emg) and (cur_data["restimulus"][start_window + offset][0] == window_label):
                offset += 1

            # Found a valid window
            #
            if offset >= self.window_size:
                emg_window = cur_data["emg"][start_window:start_window + offset]

                # Balance number of rest classes
                if (window_label == self.rest_label) and self.balance_classes:
                    if num_rest_samples > (num_samples / float(self.num_classes)):
                        #start_window += self.overlap_size
                        start_window += offset
                        continue
                    else:
                        num_rest_samples += 1
                num_samples += 1

                if obtain_all_samples:
                    win_repetition = cur_data["rerepetition"][start_window]
                    if not ((win_repetition == 2) or (win_repetition == 5)):
                        self.all_samples.append(emg_window)

                # Split into train/test:
                else:
                    win_feat = self.feature_extractor.extract_feature_point(emg_window)
                    win_repetition = cur_data["rerepetition"][start_window]

                    # Correct the window label (offsets)
                    if (window_label != self.rest_label) and adjust_labels:
                        if ex == self.E3_name:
                            window_label += self.E1_classes + self.E2_classes
                        elif ex == self.E2_name:
                            window_label += self.E1_classes

                    if (win_repetition == 2) or (win_repetition == 5):
                        self.test_features.append(win_feat)
                        self.test_labels.append(window_label)
                    else:
                        self.train_features.append(win_feat)
                        self.train_labels.append(window_label)

                # start_window += self.overlap_size
                start_window += offset

            else:
                start_window += offset

        if not obtain_all_samples:
            max_size = np.max([feature.shape[0] for feature in self.test_features] + [feature.shape[0] for feature in self.train_features])

            self.train_features = [np.concatenate([feature,np.zeros((max_size-feature.shape[0], feature.shape[1]), dtype=feature.dtype)]) if max_size > feature.shape[0] else feature for feature in self.train_features ]
            self.test_features = [np.concatenate([feature,np.zeros((max_size-feature.shape[0], feature.shape[1]), dtype=feature.dtype)]) if max_size > feature.shape[0] else feature for feature in self.test_features]


    def get_dataset_name(self):
        return "intrasubjects_variable"