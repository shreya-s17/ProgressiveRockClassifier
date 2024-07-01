import os
import numpy as np
import argparse
import fnmatch
import pydub
import concurrent.futures

from tqdm import tqdm

from custom_logging import get_logger

cd_logger = get_logger(__name__)


def create_dataset(opt):
    prog_src_root = opt['prog_src_root']
    nonprog_src_root = opt['nonprog_src_root']
    prog_dst_root = opt['prog_dst_root']
    nonprog_dst_root = opt['nonprog_dst_root']

    cd_logger.info("Sourcing audio from:")
    cd_logger.info("Non-prog:\t{}".format(prog_src_root))
    cd_logger.info("Prog:\t{}".format(nonprog_src_root))

    cd_logger.info("Saving audio to:")
    cd_logger.info("Non-prog:\t{}".format(prog_dst_root))
    cd_logger.info("Prog:\t{}".format(nonprog_dst_root))

    assert prog_src_root != prog_dst_root, cd_logger.error("Directories match! Careful.")
    assert nonprog_src_root != nonprog_dst_root, cd_logger.error("Directories match! Careful.")

    for d in (nonprog_dst_root, prog_dst_root):
        if not os.path.isdir(d):
            os.makedirs(d)

    for src_root, dst_root in ([prog_src_root, prog_dst_root],
                               [nonprog_src_root, nonprog_dst_root]):
        cd_logger.info("Populating {dst_root}...".format(dst_root=dst_root))
        paths = []
        for t_root, dirs, filenames in os.walk(src_root):
            for ffn in fnmatch.filter(filenames, '*.mp3'):
                paths.append((dst_root, t_root, ffn))

        with concurrent.futures.ProcessPoolExecutor(max_workers=opt["num_workers"]) as executor:
            futures = [executor.submit(split_audio, pack) for pack in paths]

            for future in list(tqdm(concurrent.futures.as_completed(futures), total=len(paths))):
                pass

        cd_logger.info("Done")


def split_audio(pack):
    dst_root, t_root, ffn = pack
    full_path = os.path.join(t_root, ffn)
    base_path = '.'.join(ffn.split('.')[:-1])
    x = pydub.AudioSegment.from_mp3(full_path)
    group_by = 4
    four_min = 1000 * 60 * group_by
    for i, idx in enumerate(np.arange(0, len(x), four_min)):
        xpg_path = os.path.join(dst_root, "{base_path}_{i}.mp3".format(base_path=base_path,
                                                                       i=i))
        if os.path.exists(xpg_path):
            continue

        sound = x[idx:min(idx + four_min, len(x))]
        diff = four_min - len(sound)
        if diff > 0:
            sound = sound + pydub.AudioSegment.silent(duration=diff)

        sound.export(xpg_path, format='mp3')


if __name__ == '__main__':
    opt = argparse.ArgumentParser()

    opt.add_argument('--prog_src_root',
                     help='root of prog .mp3 files. Sourced for audio data.',
                     required=True)
    opt.add_argument('--nonprog_src_root',
                     help='root of non-prog .mp3 files. Sourced for audio data.',
                     required=True)

    opt.add_argument('--prog_dst_root',
                     help='root of prog .mp3 files. Destination for clipped audio data.',
                     required=True)
    opt.add_argument('--nonprog_dst_root',
                     help='root of non-prog .mp3 files. Destination for clipped audio data.',
                     required=True)

    opt.add_argument('--num_workers',
                     default=4,
                     help='Number of available threads for data loading')

    opt = opt.parse_args()
    opt = vars(opt)

    create_dataset(opt)

