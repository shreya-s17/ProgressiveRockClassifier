import os
import fnmatch
import numpy as np

from custom_logging import get_logger

dlogger = get_logger(__name__)


class DatasetManifest(object):
    def __init__(self, opt, debug=False, quiet=True):
        self.manifest = []

        def pool_and_index_sources(root_prog, root_nonprog):
            temp_manifest = []
            for mp3_root in (root_prog, root_nonprog):
                for root, dirs, filenames in os.walk(mp3_root):
                    for ffn in fnmatch.filter(filenames, '*.mp3'):
                        full_path = os.path.join(root, ffn)
                        if mp3_root == root_prog:
                            target = 1.0
                        else:
                            target = 0.0

                        temp_manifest.append((full_path, target))

            return temp_manifest

        for d in (opt["root_train_prog"], opt["root_train_nonprog"],
                  opt["root_val_prog"], opt["root_val_nonprog"],
                  opt["root_test_prog"], opt["root_test_nonprog"]):
            if not os.path.exists(d):
                dlogger.exception("The directory doesn't exist:\t{}.\t"
                                  "Cannot allow to continue...".format(d))
                raise IOError("Empty source directory.")

        train_start = 0
        tmp_m = pool_and_index_sources(opt["root_train_prog"], opt["root_train_nonprog"])
        self.manifest.extend(tmp_m)
        train_end = len(tmp_m)
        train_idxs = list(np.arange(train_start, train_end))
        del tmp_m

        val_start = train_end
        tmp_m = pool_and_index_sources(opt["root_val_prog"], opt["root_val_nonprog"])
        self.manifest.extend(tmp_m)
        val_end = val_start + len(tmp_m)
        val_idxs = list(np.arange(val_start, val_end))
        del tmp_m

        test_start = val_end
        tmp_m = pool_and_index_sources(opt["root_test_prog"], opt["root_test_nonprog"])
        self.manifest.extend(tmp_m)
        test_end = test_start + len(tmp_m)
        test_idxs = list(np.arange(test_start, test_end))

        if debug:
            self.splits = {
                'train': np.random.permutation(train_idxs)[:opt["debug_dataset_size"]],
                'val': np.random.permutation(val_idxs)[:opt["debug_dataset_size"]],
                'test': np.random.permutation(test_idxs)[:opt["debug_dataset_size"]]
            }
        else:
            self.splits = {
                'train': train_idxs,
                'val': val_idxs,
                'test': test_idxs
            }

        self.n = sum([len(self.splits[s]) for s in self.splits.keys()])

        if not quiet:
            dlogger.debug('number of train songs:\t{}'.format(len(self.splits['train'])))
            dlogger.debug('number of val songs:\t{}'.format(len(self.splits['val'])))
            dlogger.debug('number of test songs:\t{}'.format(len(self.splits['test'])))

    def get(self, ix):
        return self.manifest[ix]
