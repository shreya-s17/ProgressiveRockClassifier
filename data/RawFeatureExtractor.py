import torch

from torch.autograd import Variable
from data.FeatureExtractor import FeatureExtractor


class RawFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super(RawFeatureExtractor, self).__init__()

    def extract_feature(self, x):
        return Variable(torch.from_numpy(x).type(torch.FloatTensor))
