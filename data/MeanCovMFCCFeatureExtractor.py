import torch
import numpy as np

from torch.autograd import Variable

from data.MFCCFeatureExtractor import MFCCFeatureExtractor


class MeanCovMFCCFeatureExtractor(MFCCFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super(MeanCovMFCCFeatureExtractor, self).__init__(*args, **kwargs)

    def extract_feature(self, signal):
        mfccs = self.getMFCCS(signal)
        meanMFCCS = np.mean(mfccs, axis=1)

        cov = np.cov(mfccs)
        upperCovIndicies = np.triu_indices(self.n_mfccs)
        upperCov = cov[upperCovIndicies]

        if self.normalize:
            meanMFCCS = self.normalizeInputs(meanMFCCS)
            upperCov = self.normalizeInputs(upperCov)

        inputs = np.concatenate((meanMFCCS, upperCov))

        inputs = Variable(torch.from_numpy(inputs))

        return inputs
