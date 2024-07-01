import librosa
import numpy as np
import torch

from torch.autograd import Variable

from data.FeatureExtractor import FeatureExtractor


class MFCCFeatureExtractor(FeatureExtractor):
    def __init__(self, sample_rate, n_mfccs, hop_length, n_fft, normalize=True):
        super(MFCCFeatureExtractor, self).__init__()
        self.sample_rate = sample_rate
        self.n_mfccs = n_mfccs
        self.normalize = normalize
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract_feature(self, signal):
        mfccs = self.getMFCCS(signal)

        if self.normalize:
            mfccs = self.normalizeInputs(mfccs)

        mfccs = Variable(torch.from_numpy(mfccs))

        return mfccs

    def getMFCCS(self, signal):
        stft = np.abs(librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length))
        mel = librosa.feature.melspectrogram(sr=self.sample_rate, S=stft ** 2)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=self.n_mfccs)

        return mfccs

    def normalizeInputs(self, x):
        mean = x.mean()
        std = x.std()
        z = (x - mean) / std
        return z
