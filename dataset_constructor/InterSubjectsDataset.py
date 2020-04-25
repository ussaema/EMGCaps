import os
from . import Dataset
#
# Rather than split training/testing based on movement number, this dataset splits based on patients:
#   --> Via use of NCC (Normalized Cross Correlation), data for patients 8 and 10 are the furthest away.
#
class InterSubjectsDataset(Dataset):

    def process_single_exercise(self, loaded_data, patient, ex, num_samples, num_rest_samples, obtain_all_samples,
                                    adjust_labels):
        cur_data    = loaded_data[patient][ex]
        num_emg     = cur_data["emg"].shape[0]

        # Look for possible windows of EMG data
        #
        start_window = 0

        while (start_window + self.window_size) <= num_emg:

            window_label = int(cur_data["restimulus"][start_window][0])

            offset = 0
            while (offset < self.window_size) and (cur_data["restimulus"][start_window + offset][0] == window_label):
                offset += 1

            # Found a valid window
            #
            if offset == self.window_size:
                emg_window = cur_data["emg"][start_window:start_window + self.window_size]

                # Balance number of rest classes
                if (window_label == self.rest_label) and self.balance_classes:
                    if num_rest_samples > (num_samples / float(self.num_classes)):
                        start_window += self.overlap_size
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

                    # Correct the window label
                    if (window_label != self.rest_label) and adjust_labels:
                        if ex == self.E3_name:
                            window_label += self.E1_classes + self.E2_classes
                        elif ex == self.E2_name:
                            window_label += self.E1_classes
                    if (os.path.basename(patient) == "s8") or (os.path.basename(patient) == "s10"):
                        self.test_features.append(win_feat)
                        self.test_labels.append(window_label)
                    else:
                        self.train_features.append(win_feat)
                        self.train_labels.append(window_label)

                start_window += self.overlap_size

            else:
                start_window += offset

    def get_dataset_name(self):
        return "intersubjects"