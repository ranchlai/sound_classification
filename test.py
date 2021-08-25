# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddleaudio
import numpy as np
import glob
import paddle
import tqdm
import argparse
from model import ESCModel

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a',
        '--audio_folder',
        type=str,
        required=True,
        help='audio folder as in ESC-50-master/audio')
    parser.add_argument(
        '-m',
        '--meta_file',
        type=str,
        required=True,
        help='meta file found in ESC-50-master/meta/esc50.csv')
    parser.add_argument('-d', '--device', default="gpu", help="gpu or cpu")
    args = parser.parse_args()
    paddle.set_device(args.device)
    lines = open(args.meta_file).read().split('\n')
    audio_files = glob.glob(args.audio_folder + '/*.wav')
    file2target = {
        l.split(',')[0]: int(l.split(',')[2])
        for l in lines[1:] if len(l) > 0
    }
    transform = paddleaudio.transforms.LogMelSpectrogram(
        sr=32000,
        win_length=1024,
        n_fft=1024,
        hop_length=320,
        n_mels=64,
        f_min=50,
        f_max=14000)

    fold_acc = []
    for fold in range(1, 6):
        print(f'testing fold {fold}')
        model = ESCModel(pretrained=True, fold=fold)
        model.eval()
        paddle.set_grad_enabled(False)
        preds = []
        targets = []
        model.eval()
        for file in tqdm.tqdm(audio_files):
            if file.split('/')[-1][0] != str(fold):
                continue
            s, r = paddleaudio.load(file, normal=False, sr=32000)
            s = paddle.to_tensor(s)
            x = transform(s)
            x = x.transpose((0, 2, 1))
            pred = model(x.unsqueeze(0))
            preds += [int(pred[0].argmax())]
            targets += [file2target[file.split('/')[-1]]]
        acc = np.mean(np.array(targets) == np.array(preds))
        fold_acc += [acc]
        print(f'fold {fold} acc {acc}')

    print(f'average acc across 5 folds is {np.mean(fold_acc)}')
