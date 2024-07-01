import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from custom_logging import get_logger
from EncoderDecoder.model import BaseRNN, AttentionRNN, weights_init
from util import print_cm, get_time_stamp, pickle_load

from data.DatasetManifest import DatasetManifest
from data.dataloader import MusicDataLoader, MusicDataset, ImbalancedMusicDatasetSampler
from data.WindowedMFCCFeatureExtractor import WindowedMFCCFeatureExtractor
from EncoderDecoder._unittest_train_lstm import lsfc100, check_model


s_logger = get_logger(__name__)


def sample(opt):
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])
    np.random.seed(opt['seed'])

    device = torch.device("cuda" if opt['cuda'] else 'cpu')

    dataset_manifest = pickle_load(os.path.join(opt["from_dir"], "dataset_manifest.pkl"))
    feature_extractor = pickle_load(os.path.join(opt["from_dir"], "feature_extractor.pkl"))

    opt["sample_rate"] = 44100
    opt["save_dir"] = None

    win_mfcc_val = MusicDataset(opt, 'val', feature_extractor, dataset_manifest,
                                quiet=True, pre_cache=True)
    val_loader = MusicDataLoader(win_mfcc_val,
                                 batch_size=opt["batch_size"],
                                 num_workers=opt['num_workers'])

    model = lsfc100(opt)
    from_path = os.path.join(opt["from_dir"], 'model_best.ckpt')
    if not os.path.exists(from_path):
        s_logger.error("No model checkpoint at {}!".format(from_path))
        return

    model.load_state_dict(torch.load(from_path))

    print(model)
    model = model.to(device)

    check_model(model, val_loader, device)


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('--test_val_prog',
                     required=False,
                     help='Root directory of VAL Prog. Rock MP3 files.')
    opt.add_argument('--test_val_nonprog',
                     required=False,
                     help='Root directory of VAL Non Prog. Rock MP3 files.')

    opt.add_argument('--load_dir',
                     default=None,
                     help='Directory to load pre-cached features from.')

    opt.add_argument('--from_dir',
                     required=True,
                     help='Directory to load checkpoint from.')

    opt.add_argument('--num_workers',
                     default=4,
                     help='Number of available threads for data loading')
    opt.add_argument('-b', '--batch_size',
                     default=512,
                     help='Model input batch size.')
    opt.add_argument('-s', '--seed',
                     default=9,
                     help='Model seed.')
    opt.add_argument('--cuda',
                     action='store_true',
                     default=False,
                     help='Use CUDA.')

    opt = opt.parse_args()
    opt = vars(opt)

    sample(opt)
