import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import librosa
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from custom_logging import get_logger
from util import print_cm, get_time_stamp, pickle_load

from data.TestDatasetManifest import TestDatasetManifest
from data.WindowedMFCCFeatureExtractor import WindowedMFCCFeatureExtractor
from SelfAttentionWindows._unittest_train_attwin import LSfcAttX


plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathptmx}',
                                       r'\usepackage[T1]{fontenc}',
                                       r'\usepackage[utf8]{inputenc}',
                                       r'\usepackage{pslatex}']
plt.rc('xtick', labelsize=4)
plt.rc('ytick', labelsize=4)
plt.rc('axes', labelsize=6)

s_logger = get_logger(__name__)


def sample(opt):
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])
    np.random.seed(opt['seed'])

    device = torch.device("cuda" if opt['cuda'] else 'cpu')

    test_manifest = TestDatasetManifest(opt, quiet=False)
    feature_extractor = pickle_load(os.path.join(opt["from_dir"], "feature_extractor.pkl"))

    opt["sample_rate"] = 44100
    opt["save_dir"] = None

    model = LSfcAttX(opt)
    from_path = os.path.join(opt["from_dir"], 'model_best.ckpt')
    if not os.path.exists(from_path):
        s_logger.error("No model checkpoint at {}!".format(from_path))
        return

    model.load_state_dict(torch.load(from_path))

    print(model)
    model = model.to(device)

    actuals = []
    preds = []
    sm = torch.nn.LogSoftmax(dim=1)

    # Do a random subset
    num_subset = 14
    # num_subset = 4

    fig, axes = plt.subplots(7, 2)
    # fig, axes = plt.subplots(2, 2)
    # fig.set_size_inches(8, 11)
    # fig.subplots_adjust()
    minute = 60 * opt["sample_rate"]
    seconds_per_window = int(np.floor(60. / feature_extractor.windows_per_minute))

    clss = {0: 'NON-PROG', 1: 'PROG'}

    x, y = 0, 0
    with torch.no_grad():
        for i, (full_path, target) in enumerate(np.random.permutation(test_manifest.manifest)[:num_subset]):
            s_logger.info("Start song:\t{}".format(full_path))
            x_i, _ = librosa.load(full_path, sr=opt["sample_rate"], mono=True)
            feat_tensor = feature_extractor.extract_feature(x_i)
            inputs = feat_tensor.float().unsqueeze(0).unsqueeze(0)

            seq_length = feat_tensor.size(1)
            input_percentages = torch.FloatTensor(1)
            input_percentages[0] = seq_length / float(seq_length)

            inputs = inputs.to(device)
            actuals.extend([target])

            wx, attention_weights = model(inputs, input_percentages, give_attention=True)
            attention_weights = attention_weights.squeeze(2).detach().cpu().numpy()
            probas = sm(wx)
            # probas = probas.mean(dim=1)

            probas = probas.detach().cpu().numpy()
            preds_i = np.argmax(probas, axis=1)
            s_logger.info("Pred:\t{}".format(preds_i[0]))
            preds.extend(list(preds_i))

            # Graphing
            num_minutes = int(np.ceil(float(len(x_i)) / minute))
            real_x = np.zeros(num_minutes * minute)

            real_x[:len(x_i)] = x_i
            idxes = np.random.permutation(np.arange(len(real_x)))[:1000 * num_minutes]
            idxes = np.asarray(np.sort(idxes))

            t = np.linspace(0, len(real_x) / opt["sample_rate"], num=len(real_x))
            t = np.random.permutation(t)[:1000 * num_minutes]
            t = sorted(t)
            axes[y, x].plot(t, real_x[idxes])

            for j, x_j in enumerate(np.arange(0, num_minutes * 60, seconds_per_window)):
                end = min(x_j + seconds_per_window, num_minutes * 60)
                print("Window {}-{}/{}".format(x_j, end, num_minutes * 60))
                axes[y, x].axvspan(x_j, end, color='red', alpha=float(attention_weights[0, j]), lw=0)

            axes[y, x].set_ylim((-1.5, 1.5))
            myouji = '/'.join(full_path.split('/')[-2:])
            myouji = myouji.replace('_', ' ')
            if len(myouji) > 27:
                myouji = myouji[:27]
                myouji = myouji + "..."
            title_i = "{} - Pred: {}".format(myouji, clss[preds_i[0]])
            axes[y, x].title.set_text(title_i)
            axes[y, x].title.set_fontsize(8)

            x += 1
            if x >= 2:
                x = 0
                y += 1
            # break

    # plt.show()
    fig.set_size_inches(7, 10)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05, hspace=0.5)
    fig.savefig('7x10-{}_{}.pdf'.format(''.join(opt["root_test"].split('/')[-2:]), get_time_stamp()))

    # acc = accuracy_score(actuals, preds)
    # prec = precision_score(actuals, preds)
    # rec = recall_score(actuals, preds)
    # cm = confusion_matrix(actuals, preds)
    #
    # print_cm(cm, ["nonprogR", "progR"])
    # s_logger.info("=\tAccuracy:\t{}".format(acc))
    # s_logger.info("=\tPrecision:\t{}".format(prec))
    # s_logger.info("=\tRecall:\t{}".format(rec))
    #
    # metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'cm': cm}


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('--root_test',
                     required=True,
                     help='Root directory of TEST MP3 files.')

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
