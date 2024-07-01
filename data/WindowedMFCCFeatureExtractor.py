import librosa
import numpy as np
import torch
from torch.autograd import Variable

from data.FeatureExtractor import FeatureExtractor
# from sklearn.decomposition import PCA


class WindowedMFCCFeatureExtractor(FeatureExtractor):
    def __init__(self, sample_rate, n_mfccs, hop_length, n_fft, windows_per_minute=6, normalize=True):
        super(WindowedMFCCFeatureExtractor, self).__init__()
        self.sample_rate = sample_rate
        self.n_mfccs = n_mfccs
        self.normalize = normalize
        self.hop_length = hop_length
        self.windows_per_minute = windows_per_minute
        self.n_fft = n_fft
        # self.svd = PCA(n_components=50)

    def extract_feature(self, signal):
        minute = self.sample_rate * 60
        samples_per_window = int(np.floor(minute / self.windows_per_minute))

        num_minutes = int(np.ceil(float(len(signal)) / minute))
        num_windows = num_minutes * self.windows_per_minute
        windows = np.zeros((num_windows, samples_per_window))

        # Start at samples_per_window instead of 0
        for i, x_i in enumerate(np.arange(0, len(signal), samples_per_window)):
            end = min(x_i + samples_per_window, len(signal))
            windows[i, :end - x_i] = signal[x_i: end]

        inputs = []
        for i, w_i in enumerate(windows):
            mfccs = self.getMFCCS(w_i)
            meanMFCCS = np.mean(mfccs, axis=1)

            # cov = np.cov(mfccs)
            # upperCovIndicies = np.triu_indices(self.n_mfccs)
            # upperCov = cov[upperCovIndicies]

            if self.normalize:
                meanMFCCS = self.normalizeInputs(meanMFCCS)
                # upperCov = self.normalizeInputs(upperCov)
                # mfccs = self.normalizeInputs(mfccs)

            # in_i = np.concatenate((meanMFCCS, upperCov))

            in_i = meanMFCCS
            inputs.append(in_i)

        inputs = np.asarray(inputs)

        # /!\
        # inputs = self.svd.fit_transform(inputs)

        inputs = torch.from_numpy(inputs).transpose(0, 1)
        inputs = Variable(inputs)

        return inputs

    def getMFCCS(self, signal):
        # stft = np.abs(librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length))
        # mel = librosa.feature.melspectrogram(sr=self.sample_rate, S=stft ** 2)
        mel = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=self.n_mfccs)

        return mfccs

    def normalizeInputs(self, x):
        mean = x.mean()
        std = x.std()
        z = (x - mean) / std
        return z