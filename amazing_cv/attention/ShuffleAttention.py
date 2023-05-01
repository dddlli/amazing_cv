import numpy as np
import paddle
import paddle.nn as nn
from ..utils.init import *


class ShuffleAttention(nn.Layer):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = paddle.create_parameter(shape=[1, channel // (2 * G), 1, 1], dtype='float32')
        print(self.cweight.shape)
        self.cbias = paddle.create_parameter(shape=[1, channel // (2 * G), 1, 1], dtype='float32')
        self.sweight = paddle.create_parameter(shape=[1, channel // (2 * G), 1, 1], dtype='float32')
        self.sbias = paddle.create_parameter(shape=[1, channel // (2 * G), 1, 1], dtype='float32')
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.named_sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                ones_(m.weight)
                zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                normal_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape([b, groups, -1, h, w])
        x = x.transpose([0, 2, 1, 3, 4])

        # flatten
        x = x.reshape([b, -1, h, w])

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        # group into subfeatures
        x = x.reshape([b * self.G, -1, h, w])  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, axis=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = paddle.concat([x_channel, x_spatial], axis=1)  # bs*G,c//G,h,w
        out = out.reshape([b, -1, h, w])

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


if __name__ == '__main__':
    input = paddle.randn([50, 512, 7, 7])
    se = ShuffleAttention(channel=512, G=8)
    output = se(input)
    print(output.shape)
