from . import FeatureExtractor
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pylab
import librosa

class Spectrograms(FeatureExtractor):
    def create_spectrogram(self, data, fft_size= 20, fs= 2000, overlap_fac= 0.5):
        hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
        pad_end_size = fft_size  # the last segment can overlap the end of the data array by no more than one window size
        total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))
        t_max = len(data) / np.float32(fs)

        window = np.hanning(fft_size)  # our half cosine window
        inner_pad = np.zeros(fft_size)  # the zeros which will be used to double each segment size

        proc = np.concatenate((data, np.zeros(pad_end_size)))  # the data to process
        result = np.empty((total_segments, fft_size), dtype=np.float32)  # space to hold the result

        for i in range(total_segments):  # for each segment
            current_hop = hop_size * i  # figure out the current segment offset
            segment = proc[current_hop:current_hop + fft_size]  # get the current segment
            windowed = segment * window  # multiply by the half cosine function
            padded = np.append(windowed, inner_pad)  # add 0s to double the length of the data
            spectrum = np.fft.fft(padded) / fft_size  # take the Fourier Transform and scale by the number of samples
            autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
            result[i, :] = autopower[:fft_size]  # append to the results array

        result = 20 * np.log10(result)  # scale to db
        return np.clip(result, -40, 200)  # clip values
    def extract_feature_point(self, raw_samples):
        #specs = np.array([self.create_spectrogram(raw_samples[:, i], fs=2000) for i in range(len(raw_samples[1, :]))])
        #fig = plt.figure()
        #plt.imshow(specs[0], origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
        #plt.show()
        return raw_samples #np.expand_dims(raw_samples[:,1], 1)

    def global_setup(self, all_raw_samples):
        pass