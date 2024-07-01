import librosa
import numpy as np
import scipy.signal
import torch

from data.FeatureExtractor import FeatureExtractor


windows = {'hamming': scipy.signal.hamming,
           'hann': scipy.signal.hann,
           'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


class STFTFeatureExtractor(FeatureExtractor):
    def __init__(self, sample_rate=44100, normalize=True):
        super(STFTFeatureExtractor, self).__init__()

        # Some params borrowed from DeepSpeech project : https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py
        # Sample rate 44100 for music (voice is 16000)
        self.sample_rate = sample_rate
        self.window_size = 0.02  # Window size for spectrogram in seconds
        self.window_stride = 0.01  # Window stride for spectrogram in seconds
        self.n_fft = int(sample_rate * self.window_size)
        self.window = windows['hamming']  # Window type for spectrogram generation
        self.normalize = normalize

    def extract_feature(self, x):
        num_sec = float(len(x)) / self.sample_rate
        num_min = int(np.ceil(float(num_sec) / 60.))
        # Taking the whole music sample runs out of memory (10 minute ~= 6GB RAM).
        # Instead, concat some small 10 second windows. Take 10 seconds @ 16khz from every minute of audio
        xp = np.zeros((self.sample_rate * num_min * 60))
        xp[:len(x)] = x
        xp = xp.reshape(num_min, self.sample_rate * 60)
        # sub-sample minute-wise. 10 seconds per minute, from end to avoid intro
        xp = xp[:, -self.sample_rate * 10:]
        xp = np.concatenate(xp, axis=0)
        x = xp

        # Short-time Fourier transform (STFT) spectrogram (complex valued)
        window_length = self.n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        n_fft = self.n_fft

        D = librosa.stft(x, n_fft=n_fft, hop_length=hop_length,
                         win_length=window_length, window=self.window)

        # Separate a complex-valued spectrogram D into its magnitude (S) and phase (P) components, so that D = S * P.
        spect, phase = librosa.magphase(D)

        # S = log(S + 1)
        # Return the natural logarithm of one plus the input array, element-wise.
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect
