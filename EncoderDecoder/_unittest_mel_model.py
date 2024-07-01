import os
import argparse
import torch

from torch.utils.data import DataLoader
from data.dataloader import *
from data.MFCCFeatureExtractor import MFCCFeatureExtractor
from data.MeanCovMFCCFeatureExtractor import MeanCovMFCCFeatureExtractor
from data.STFTFeatureExtractor import STFTFeatureExtractor


def model_test(opt):
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])
    np.random.seed(opt['seed'])

    device = torch.device("cuda" if opt['cuda'] else 'cpu')

    mel_feature_extractor = MeanCovMFCCFeatureExtractor(opt['sample_rate'],
                                                        n_mfccs=20,
                                                        hop_length=512,
                                                        n_fft=2048)
    dataset_mel = MusicDataset(opt, 'train', mel_feature_extractor,
                               debug=True, quiet=True, pre_cache=True)

    loader = DataLoader(dataset_mel, batch_size=opt["batch_size"], shuffle=True, num_workers=opt['num_workers'])

    for data in loader:
        x = data['x']
        label = data['gt']
        print(x.shape)


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('--root_prog',
                     required=True,
                     help='Root directory of Prog. Rock MP3 files.')
    opt.add_argument('--root_nonprog',
                     required=True,
                     help='Root directory of Non Prog. Rock MP3 files.')
    opt.add_argument('--num_hidden',
                     default=800,
                     help='Number of hidden units in recurrent layers')
    opt.add_argument('--latent_size',
                     default=100,
                     help='Number of latent features for final layer')
    opt.add_argument('--num_workers',
                     default=4,
                     help='Number of available threads for data loading')
    opt.add_argument('-b', '--batch_size',
                     default=2,
                     help='Model input batch size.')
    opt.add_argument('-s', '--seed',
                     default=9,
                     help='Model seed.')
    opt.add_argument('--cuda',
                     action='store_true',
                     default=False,
                     help='Use CUDA.')
    opt.add_argument('--debug_dataset_size',
                     default=32,
                     help='Size of debug dataset (n)')
    opt.add_argument('--sample_rate',
                     default=44100,
                     help='Sample rate to load and process music files.')
    opt = opt.parse_args()
    opt = vars(opt)

    model_test(opt)
