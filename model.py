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

import paddle.nn as nn
import os
import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.utils.download import get_weights_path_from_url

urls = [
    'https://bj.bcebos.com/paddleaudio/models/esc50/esc50_fold1_test_acc_0.932.pd',
    'https://bj.bcebos.com/paddleaudio/models/es50/esc50_fold3_test_acc_0.948.pd',
    'https://bj.bcebos.com/paddleaudio/models/es50/esc50_fold2_test_acc_0.968.pd',
    'https://bj.bcebos.com/paddleaudio/models/es50/esc50_fold4_test_acc_0.955.pd',
    'https://bj.bcebos.com/paddleaudio/models/es50/esc50_fold5_test_acc_0.932.pd',
]


class ConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias_attr=False)

        self.conv2 = nn.Conv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias_attr=False)

        self.bn1 = nn.BatchNorm2D(out_channels)
        self.bn2 = nn.BatchNorm2D(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn14(nn.Layer):
    def __init__(self, ):

        super(Cnn14, self).__init__()

        self.bn0 = nn.BatchNorm2D(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias_attr=True)
        self.fc_audioset = nn.Linear(2048, 527, bias_attr=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = x.transpose([0, 3, 2, 1])
        x = self.bn0(x)
        x = x.transpose([0, 3, 2, 1])

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.mean(axis=3, keepdim=True)
        x1 = x.max(axis=2, keepdim=True)
        x2 = x1.mean(axis=2, keepdim=True)
        x = x1 + x2
        x = x.squeeze()
        x = x.unsqueeze(0)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        return embedding


def get_checkpoint():
    if not os.path.exists(c.audioset_checkpoint):
        os.makedirs(os.path.split(c.audioset_checkpoint)[0], exist_ok=True)
        print('Downloading audioset checkpoint...')
        os.system(
            'wget https://paddlenlp.bj.bcebos.com/models/Cnn14_class=527mAP=0.431.pd.tar -o {}'.
            format(c.audioset_checkpoint))
        print('done')


class ESCModel(nn.Layer):
    def __init__(self, pretrained=True, fold=1):
        super(ESCModel, self).__init__()
        assert isinstance(fold, int) and fold <= 5 and fold >= 1
        self.audioset_model = Cnn14()
        self.fc_esc50 = nn.Linear(2048, 50, bias_attr=True)
        self.drop = nn.Dropout(0.5)
        if pretrained:
            path = get_weights_path_from_url(urls[fold - 1])
            self.load_dict(paddle.load(path))

    def forward(self, X):
        out = self.audioset_model(X)
        out = self.drop(out)
        logits = self.fc_esc50(out)
        return logits
