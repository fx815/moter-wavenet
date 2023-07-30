# coding: utf-8
"""
Make subset of dataset

usage: mksubset.py [options] <in_dir> <out_dir>

options:
    -h, --help               Show help message.
    --limit=<N>              Limit dataset size by N-hours [default: 10000].
    --train-dev-test-split   Train/test split.
    --dev-size=<N>           Development size or rate [default: 0.1].
    --test-size=<N>          Test size or rate [default: 0.1].
    --target-sr=<N>          Resampling.
    --random-state=<N>       Random seed [default: 1234].
"""
from docopt import docopt
import librosa
from glob import glob
from os.path import join, basename, exists, splitext
from tqdm import tqdm
import sys
import os
from shutil import copy2
from scipy.io import wavfile
import numpy as np


def read_wav_or_raw(src_file, is_raw):
    if is_raw:
        sr = 24000  # hard coded for now
        x = np.fromfile(src_file, dtype=np.int16)
    else:
        sr, x = wavfile.read(src_file)
    return sr, x


def write_wav_or_raw(dst_path, sr, x, is_raw):
    if is_raw:
        x.tofile(dst_path)
    else:
        wavfile.write(dst_path, sr, x)

if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    limit = float(args["--limit"])
    train_dev_test_split = args["--train-dev-test-split"]
    dev_size = float(args["--dev-size"])
    test_size = float(args["--test-size"])
    target_sr = args["--target-sr"]
    target_sr = int(target_sr) if target_sr is not None else None
    random_state = int(args["--random-state"])

    src_files = sorted(glob(join(in_dir, "*wave.npy")))
    if len(src_files) == 0:
        raise RuntimeError("No files found in {}".format(in_dir))

    total_samples = 0
    indices = []
    file_names = []
    signed_int32_max = 2**31

    os.makedirs(out_dir, exist_ok=True)
    if train_dev_test_split:
        os.makedirs(join(out_dir, "train_no_dev"), exist_ok=True)
        os.makedirs(join(out_dir, "dev"), exist_ok=True)
        os.makedirs(join(out_dir, "eval"), exist_ok=True)

    print("Total number of utterances: {}".format(len(src_files)))
    for idx, src_file in tqdm(enumerate(src_files)):
        indices.append(idx)
    print(src_files)

    if train_dev_test_split:
        train_indices=[2,3,4]
        dev_indices = [0]
        eval_indices = [1]

        sets = [
            (sorted(train_indices), join(out_dir, "train_no_dev")),
            (sorted(dev_indices), join(out_dir, "dev")),
            (sorted(eval_indices), join(out_dir, "eval")),
        ]
    else:
        sets = [(indices, out_dir)]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    total_samples = {}
    sr = 48000
    for indices, d in sets:
        set_name = basename(d)
        total_samples[set_name] = 0
        for idx in tqdm(indices):
            src_file = src_files[idx]
            dst_path = join(d, basename(src_file))
            x = np.load(src_file)
            is_int32 = x.dtype == np.int32
            if is_int32:
                x = x.astype(np.float64) / signed_int32_max
            scaler.partial_fit(x.astype(np.float64).reshape(-1, 1))
            total_samples[set_name] += len(x)
            copy2(src_file, dst_path)
            copy2(src_file.replace('wave.npy','feats.npy'),dst_path.replace('wave.npy','feats.npy'))

    print("Waveform min: {}".format(scaler.data_min_))
    print("Waveform max: {}".format(scaler.data_max_))
    absmax = max(np.abs(scaler.data_min_[0]), np.abs(scaler.data_max_[0]))
    print("Waveform absolute max: {}".format(absmax))
    if absmax > 1.0:
        print("There were clipping(s) in your dataset.")
    print("Global scaling factor would be around {}".format(1.0 / absmax))

    if train_dev_test_split:
        print("Train/dev/test split:")
        for n, s in zip(["train_no_dev", "dev", "eval"], sets):
            hours = total_samples[n] / sr / 3600.0
            print("{}: {:.2f} hours ({} utt)".format(n, hours, len(s[0])))

    sys.exit(0)
