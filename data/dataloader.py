import os
import numpy as np
import torch
import scipy.signal
import librosa
import fnmatch
import concurrent.futures
import torch.multiprocessing

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from data.ImbalancedDatasetSampler import ImbalancedDatasetSampler
from global_constants import *
from custom_logging import get_logger
from util import pickle_write, pickle_load


dlogger = get_logger(__name__)


class MusicDataset(Dataset):
    def __init__(self, opt, mode, feature_extractor, dataset_manifest, quiet=True, pre_cache=False):
        super(MusicDataset, self).__init__()
        self.opt = opt
        self.mode = mode  # to load train/val/test data
        self.quiet = quiet
        # load the json file which contains information about the dataset
        self.dataset_manifest = dataset_manifest
        self.dataset_cache = {}
        self.feature_extractor = feature_extractor
        self.sample_rate = opt["sample_rate"]
        self.save_path = None
        self.load_path = None

        save = opt["save_dir"]
        load = opt["load_dir"]

        if load is not None:
            self.load_path = os.path.join(load, "{}.pkl".format(self.mode))
            if not os.path.exists(self.load_path):
                self.load_path = None
                dlogger.warn("Given load_dir does not exist! Skipping load sequence.")
            else:
                dlogger.info("Load features from:\t{}".format(self.load_path))
                self.dataset_cache = pickle_load(self.load_path)
        elif pre_cache:
            dlogger.info("Pre-caching features...")
            jobs = []
            for ix in self.dataset_manifest.splits[self.mode]:
                full_path, target = self.dataset_manifest.get(ix)
                jobs.append((ix, full_path))

            with concurrent.futures.ProcessPoolExecutor(max_workers=opt["num_workers"]) as executor:
                futures = [executor.submit(self.cache_feat_job, job) for job in jobs]
                for future in list(tqdm(concurrent.futures.as_completed(futures), total=len(jobs))):
                    ix, feat = future.result()
                    self.dataset_cache[ix] = feat

            if save is not None:
                # TODO(WG): Save params that describe feature extractor
                self.save_path = os.path.join(save, "{}.pkl".format(self.mode))
                if not os.path.isdir(save):
                    os.makedirs(save)

                dlogger.info("Save features to:\t{}".format(self.save_path))
                pickle_write(self.save_path, self.dataset_cache)

        dlogger.info("Finished initializing dataloader.")

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = ix % len(self.dataset_manifest.splits[self.mode])
        ix = self.dataset_manifest.splits[self.mode][ix]

        full_path, target = self.dataset_manifest.get(ix)
        if not self.quiet: dlogger.debug('Load ix={}'.format(ix))

        data = {}
        if self.dataset_cache.get(ix) is not None:
            # dlogger.debug("hit cache {ix}".format(ix=ix))
            data['x'] = self.dataset_cache[ix]
        else:
            signal = self.load_audio(full_path)
            feat = self.feature_extractor.extract_feature(signal)
            self.dataset_cache[ix] = feat
            data['x'] = feat

        data['path'] = full_path
        data['gt'] = Variable(torch.tensor(float(target)))
        data['sample_id'] = ix

        return data

    def __len__(self):
        return len(self.dataset_manifest.splits[self.mode])

    def load_audio(self, full_path):
        x, _ = librosa.load(full_path, sr=self.sample_rate, mono=True)
        return x

    def cache_feat_job(self, meta):
        ix, fpath = meta
        if not self.quiet: dlogger.debug('Load ix={}'.format(ix))
        signal = self.load_audio(fpath)
        feat = self.feature_extractor.extract_feature(signal)
        return ix, feat


def _collate_fn(batch):
    """
    Modified version of DS2 collate
    :param batch: The mini batched data from dataset __getitem__
    :return:
    """

    def func(p):
        return p['x'].size(1)

    batch = sorted(batch, key=lambda sample: sample['x'].size(1), reverse=True)
    longest_sample = max(batch, key=func)['x']
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    # Pseudo-image shape - BxCxDxT: Batch x Channel (1 channel) x Freq bins x Time bins
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample['x']
        target = sample['gt']
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        targets.append(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages


class ImbalancedMusicDatasetSampler(ImbalancedDatasetSampler):
    def __init__(self, *args, **kwargs):
        super(ImbalancedMusicDatasetSampler, self).__init__(*args, **kwargs)

    def _get_label(self, dataset, idx):
        if type(dataset) is MusicDataset:
            return int(dataset.__getitem__(idx)['gt'].int())
        else:
            raise NotImplementedError


class MusicDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Data loader for MusicDataset to replace collate_fn
        """
        super(MusicDataLoader, self).__init__(*args, **kwargs)

        self.collate_fn = _collate_fn
