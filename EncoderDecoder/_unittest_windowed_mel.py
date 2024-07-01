import argparse
import torch
import torch.optim
import numpy as np
from data.WindowedMFCCFeatureExtractor import WindowedMFCCFeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from data.DatasetManifest import DatasetManifest
from data.dataloader import MusicDataLoader, MusicDataset
from EncoderDecoder.model import ConvED


opt = {"root_prog": "/media/wgar/New Volume1/dataset/cap6610sp19_project/Progressive Rock Songs/",
       "root_nonprog": "/media/wgar/New Volume1/dataset/cap6610sp19_project/0_Not_prog/",
       "num_workers": 1,
       "batch_size": 2,
       "seed": 9,
       "cuda": True,
       "debug_dataset_size": 32,
       "sample_rate": 44100
      }

torch.manual_seed(opt['seed'])
torch.cuda.manual_seed_all(opt['seed'])
np.random.seed(opt['seed'])

device = torch.device("cuda" if opt['cuda'] else 'cpu')

mfcc_feature_extractor = WindowedMFCCFeatureExtractor(opt["sample_rate"], 20, 512, 2048, 12)
dataset_manifest = DatasetManifest(opt, debug=True, quiet=False)
mel_train = MusicDataset(opt, 'train', mfcc_feature_extractor, dataset_manifest,
                              quiet=False, pre_cache=False)
train_loader = MusicDataLoader(mel_train,
                               batch_size=opt["batch_size"],
                               shuffle=True,
                               num_workers=opt['num_workers'])

for data in train_loader:
    print(data[0].shape)
    print(data[1])
    print(data[2])
    # break
