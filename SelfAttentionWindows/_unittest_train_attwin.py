import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np

from custom_logging import get_logger
from EncoderDecoder._unittest_train_lstm import check_model
from util import print_cm, get_time_stamp, pickle_write

from data.DatasetManifest import DatasetManifest
from data.dataloader import MusicDataLoader, MusicDataset, ImbalancedMusicDatasetSampler
from data.WindowedMFCCFeatureExtractor import WindowedMFCCFeatureExtractor
from SelfAttentionWindows.architectures import *

t_logger = get_logger(__name__)


def model_test(opt):
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])
    np.random.seed(opt['seed'])

    device = torch.device("cuda" if opt['cuda'] else 'cpu')

    # feature_extractor = WindowedSTFTFeatureExtractor(opt["sample_rate"], 20,
    #                                                  hop_length=1024, n_fft=512, windows_per_minute=5)
    feature_extractor = WindowedMFCCFeatureExtractor(opt["sample_rate"], 20, 512, 2048, windows_per_minute=4)
    dataset_manifest = DatasetManifest(opt, debug=opt["debug"], quiet=False)

    win_mfcc_train = MusicDataset(opt, 'train', feature_extractor, dataset_manifest,
                                  quiet=True, pre_cache=True)
    train_sampler = ImbalancedMusicDatasetSampler(win_mfcc_train)
    train_loader = MusicDataLoader(win_mfcc_train,
                                   sampler=train_sampler,
                                   batch_size=opt["batch_size"],
                                   num_workers=opt['num_workers'])

    win_mfcc_val = MusicDataset(opt, 'val', feature_extractor, dataset_manifest,
                                quiet=True, pre_cache=True)
    val_loader = MusicDataLoader(win_mfcc_val,
                                 batch_size=opt["batch_size"],
                                 num_workers=opt['num_workers'])

    win_mfcc_test = MusicDataset(opt, 'test', feature_extractor, dataset_manifest,
                                 quiet=True, pre_cache=True)
    test_loader = MusicDataLoader(win_mfcc_test,
                                  batch_size=opt["batch_size"],
                                  shuffle=True,
                                  num_workers=opt['num_workers'])

    if opt["db_gen_only"]:
        return

    t_logger.info('Architecture:\t{}'.format(opt['arch']))

    if opt['arch'] == 'ED':
        model = lsfc100(opt)
    elif opt['arch'] == 'ED+Res':
        model = lsfc100res(opt)
    elif opt['arch'] == 'ED+Res+Att':
        model = LSfcAttX(opt)
    else:
        raise ValueError("Non-existant architecture:\t{}".format(opt['arch']))

    print(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    print(optimizer)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 120, 180], gamma=0.5)
    print(scheduler)

    if opt["ckpt_dir"]:
        ckpt_dir = os.path.join(opt["ckpt_dir"], get_time_stamp())
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)

        t_logger.info("Storing dataset_manifest\t->\t{}".format(ckpt_dir))
        pickle_write(os.path.join(ckpt_dir, "dataset_manifest.pkl"), dataset_manifest)

        t_logger.info("Storing feature_extractor\t->\t{}".format(ckpt_dir))
        pickle_write(os.path.join(ckpt_dir, "feature_extractor.pkl"), feature_extractor)

        t_logger.info("Checkpointing to {}.".format(ckpt_dir))
    else:
        ckpt_dir = None

    # The target that this loss expects is a class index (0 to C-1, where C = number of classes)
    crit = torch.nn.NLLLoss()
    sm = torch.nn.LogSoftmax(dim=1)

    macro_history = []
    metric_history = []
    acc_history = []
    for epoch in range(opt["epochs"]):
        scheduler.step()
        micro_history = []
        for i, data in enumerate(train_loader):
            inputs, targets, input_percentages = data

            inputs = inputs.to(device)
            targets = targets.to(device).long()
            # print(targets)

            wx = model(inputs, input_percentages)
            probas = sm(wx)
            # Average over windows
            # probas = probas.mean(dim=1)

            loss = crit(probas, targets)
            micro_history.append(loss.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # t_logger.info("[{epoch}][{i}]\tLoss\t:\t{loss:.4f}".format(epoch=epoch, i=i, loss=loss))

        avg_loss = np.asarray(micro_history).mean()
        # Check so far
        t_logger.info("===\tEnd of epoch {epoch}\t=== Avg. Loss:\t{L_bar}".format(epoch=epoch, L_bar=avg_loss))
        metrics_e = check_model(model, val_loader, device, return_metrics=True)

        macro_history.append(avg_loss)
        metric_history.append(metrics_e)
        acc_e = metrics_e['accuracy']
        acc_history.append(acc_e)

        if ckpt_dir:
            if epoch > 0 and acc_e > np.max(acc_history[:-1]):
                # Checkpoint best_so_far
                t_logger.info("Best so far; Checkpoint model\t->\t{}".format(ckpt_dir))
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_best.ckpt"))

            if epoch % opt["save_freq"] == 0:
                # Checkpoint
                t_logger.info("Checkpoint model\t->\t{}".format(ckpt_dir))

                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_{}.ckpt".format(epoch)))

            # Update meta info
            pickle_write(os.path.join(ckpt_dir, "macro_history.pkl"), macro_history)
            pickle_write(os.path.join(ckpt_dir, "metric_history.pkl"), metric_history)
            pickle_write(os.path.join(ckpt_dir, "acc_history.pkl"), acc_history)

    check_model(model, test_loader, device)


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('--root_train_prog',
                     required=True,
                     help='Root directory of TRAIN Prog. Rock MP3 files.')
    opt.add_argument('--root_train_nonprog',
                     required=True,
                     help='Root directory of TRAIN Non Prog. Rock MP3 files.')

    opt.add_argument('--root_val_prog',
                     required=True,
                     help='Root directory of VAL Prog. Rock MP3 files.')
    opt.add_argument('--root_val_nonprog',
                     required=True,
                     help='Root directory of VAL Non Prog. Rock MP3 files.')

    opt.add_argument('--root_test_prog',
                     required=True,
                     help='Root directory of TEST Prog. Rock MP3 files.')
    opt.add_argument('--root_test_nonprog',
                     required=True,
                     help='Root directory of TEST Non Prog. Rock MP3 files.')

    opt.add_argument('--arch',
                     choices=['ED+Res+Att', 'ED+Res', 'ED'],
                     default='ED+Res+Att')
    opt.add_argument('--save_dir',
                     default=None,
                     help='Directory to store model pre-cached features.')
    opt.add_argument('--load_dir',
                     default=None,
                     help='Directory to load pre-cached features from.')
    opt.add_argument('--ckpt_dir',
                     default=None,
                     help='Directory load/save model checkpoints.')
    opt.add_argument('--save_freq',
                     default=20,
                     type=int,
                     help='Epochs to wait before checkpoint the model.')

    opt.add_argument('--epochs',
                     default=500,
                     help='Number of training epochs.')
    opt.add_argument('--hidden_size',
                     default=512,
                     help='Number of hidden units in recurrent layers')
    opt.add_argument('--latent_size',
                     default=1024,
                     help='Number of latent features for final layer')
    opt.add_argument('--num_workers',
                     default=4,
                     help='Number of available threads for data loading')
    opt.add_argument('-b', '--batch_size',
                     default=256,
                     help='Model input batch size.')
    opt.add_argument('-s', '--seed',
                     default=9,
                     help='Model seed.')
    opt.add_argument('--cuda',
                     action='store_true',
                     default=False,
                     help='Use CUDA.')
    opt.add_argument('--sample_rate',
                     default=44100,
                     help='Sample rate to load and process music files.')
    opt.add_argument('--debug',
                     default=False,
                     action='store_true',
                     help='Debug mode.')
    opt.add_argument('--debug_dataset_size',
                     default=4,
                     help='Size of debug dataset (n)',
                     type=int)
    opt.add_argument('--db_gen_only',
                     help='If set, generate dataset files and exit.',
                     action='store_true',
                     default=False)

    opt = opt.parse_args()
    opt = vars(opt)

    model_test(opt)
