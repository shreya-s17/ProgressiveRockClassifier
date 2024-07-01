import os
import argparse
import fnmatch
import tqdm
import subprocess

from global_constants import *
from custom_logging import get_logger
from multiprocessing import Pool as ThreadPool
from ffmpy import FFmpeg

process_logger = get_logger(__name__)


def process_audio_file(job_params):
    """
    Must have ffmpeg version >= 4.1.1
    :param job_params:
    :return:
    """
    try:
        from_path, proc_path = job_params
        ff = FFmpeg(inputs={from_path: None}, outputs={proc_path: encode_params})
        _, err_str = ff.run(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        process_logger.exception(e)


def process_audio_tree(opt):
    base_dir = opt.collection_dir
    if not base_dir.endswith('/'):
        base_dir = base_dir + '/'

    out_dir = opt.out_dir

    if not os.path.exists(base_dir):
        process_logger.exception("No directory found at:\t{}".format(base_dir))
        return

    if not os.path.exists(out_dir):
        process_logger.info("Create directory at:\t{}".format(out_dir))
        os.makedirs(out_dir)

    jobs = []

    for root, dir, filenames in os.walk(base_dir):
        for pattern in from_patterns:
            for fname in fnmatch.filter(filenames, pattern):
                from_path = os.path.join(root, fname)
                proc_fname = ''.join(fname.split('.')[:-1] + [base_music_format])
                append_root = root[len(base_dir):]
                proc_base_dir = os.path.join(out_dir, append_root)
                proc_path = os.path.join(proc_base_dir, proc_fname)
                if os.path.exists(proc_path):
                    continue

                process_logger.debug("{}\t->\t{}".format(from_path, proc_path))
                if not os.path.exists(proc_base_dir):
                    os.makedirs(proc_base_dir)

                jobs.append([from_path, proc_path])

    threadPool = ThreadPool(4)
    for _ in tqdm.tqdm(threadPool.imap_unordered(process_audio_file, jobs), total=len(jobs)):
        pass


if __name__ == '__main__':
    pat_opt = argparse.ArgumentParser()

    pat_opt.add_argument("--collection_dir", "-cd",
                         help="Root directory to audio files to process.",
                         required=True)
    pat_opt.add_argument("--out_dir", "-od",
                         help="Output directory of output audio files.",
                         required=True)

    pat_opt, _ = pat_opt.parse_known_args()

    process_audio_tree(pat_opt)
