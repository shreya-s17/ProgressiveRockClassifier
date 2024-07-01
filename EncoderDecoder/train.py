import os
import argparse
import torch
import torch.optim
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score

from custom_logging import get_logger
from data.DatasetManifest import DatasetManifest
from data.dataloader import MusicDataLoader, MusicDataset
from data.STFTFeatureExtractor import STFTFeatureExtractor
from EncoderDecoder.model import ConvED

t_logger = get_logger(__name__)

debug = True


def model_test(opt):
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])
    np.random.seed(opt['seed'])

    device = torch.device("cuda" if opt['cuda'] else 'cpu')

    stft_feature_extractor = STFTFeatureExtractor()
    dataset_manifest = DatasetManifest(opt, debug=debug)

    stft_train = MusicDataset(opt, 'train', stft_feature_extractor, dataset_manifest,
                              quiet=True, pre_cache=True)
    train_loader = MusicDataLoader(stft_train,
                                   batch_size=opt["batch_size"],
                                   shuffle=True,
                                   num_workers=opt['num_workers'])

    stft_val = MusicDataset(opt, 'val', stft_feature_extractor, dataset_manifest,
                            quiet=True, pre_cache=True)
    val_loader = MusicDataLoader(stft_val,
                                 batch_size=opt["batch_size"],
                                 shuffle=True,
                                 num_workers=opt['num_workers'])

    model = ConvED(opt)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # The target that this loss expects is a class index (0 to C-1, where C = number of classes)
    crit = torch.nn.NLLLoss()
    sm = torch.nn.LogSoftmax(dim=1)

    for epoch in range(opt["epochs"]):
        for i, data in enumerate(train_loader):
            inputs, targets, input_percentages = data
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            print(targets)

            wx = model(inputs, input_percentages)
            probas = sm(wx)
            loss = crit(probas, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_logger.info("[{epoch}][{i}]\tLoss\t:\t{loss:.4f}".format(epoch=epoch, i=i, loss=loss))

        # Check so far
        t_logger.info("===\tEnd of epoch {epoch}\t===".format(epoch=epoch))
        check_model(model, val_loader, device)

    stft_test = MusicDataset(opt, 'test', stft_feature_extractor, dataset_manifest,
                             quiet=True, pre_cache=True)
    test_loader = MusicDataLoader(stft_test,
                                  batch_size=opt["batch_size"],
                                  shuffle=True,
                                  num_workers=opt['num_workers'])

    check_model(model, test_loader, device)


def check_model(model, x_loader, device):
    actuals = []
    preds = []
    sm = torch.nn.LogSoftmax(dim=1)

    with torch.no_grad():
        for i, data in enumerate(x_loader):
            inputs, targets, input_percentages = data
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            actuals.extend(list(targets.cpu().numpy().astype(int)))

            wx = model(inputs, input_percentages)
            probas = sm(wx)
            probas = probas.detach().cpu().numpy()
            preds_i = np.argmax(probas, axis=1)
            preds.extend(list(preds_i))

    acc = accuracy_score(actuals, preds)
    prec = precision_score(actuals, preds)
    rec = recall_score(actuals, preds)

    t_logger.info("=\tAccuracy:\t{}".format(acc))
    t_logger.info("=\tPrecision:\t{}".format(prec))
    t_logger.info("=\tRecall:\t{}".format(rec))


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('--root_prog',
                     required=True,
                     help='Root directory of Prog. Rock MP3 files.')
    opt.add_argument('--root_nonprog',
                     required=True,
                     help='Root directory of Non Prog. Rock MP3 files.')
    opt.add_argument('--epochs',
                     default=5,
                     help='Number of training epochs.')
    opt.add_argument('--hidden_size',
                     default=512,
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
