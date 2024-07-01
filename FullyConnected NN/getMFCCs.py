
import numpy as np
import librosa

audioPath  = "/Users/nkroeger/Downloads/DownloadedSongs/Fantasy.mp3"
numOfMFCCS = 20
durationToLoad = 5.0 #given in seconds

def getMFCCS(x, sr, nMFCCs):
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    meanMFCCS = np.mean(mfccs, axis=1)

    cov = np.cov(mfccs)
    upperCovIndicies = np.triu_indices(nMFCCs)
    upperCov = cov[upperCovIndicies]
    return meanMFCCS, upperCov, mfccs

def normalizeInputs(x):
    mean = x.mean()
    std = x.std()
    z = (x-mean)/std
    return z

x, sr = librosa.load(audioPath, sr=None, mono=True, duration=durationToLoad)
meanMFCC, covMFCC, mfccs = getMFCCS(x, sr, numOfMFCCS)
meanMFCCNormalized = normalizeInputs(meanMFCC)
covMFCCNormalized = normalizeInputs(covMFCC)
