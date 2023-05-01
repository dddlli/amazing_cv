import numpy as np
import paddle
from paddle import nn


def spatial_shift1(x):
    b, w, h, c = x.shape
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.shape
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class SplitAttention(nn.Layer):
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias_attr=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias_attr=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape([b, k, -1, c])  # bs,k,n,c
        a = paddle.sum(paddle.sum(x_all, 1), 1)  # bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # bs,kc
        hat_a = hat_a.reshape([b, self.k, c])  # bs,k,c
        bar_a = self.softmax(hat_a)  # bs,k,c
        attention = bar_a.unsqueeze(-2)  # #bs,k,1,c
        out = attention * x_all  # #bs,k,n,c
        out = paddle.sum(out, 1).reshape([b, h, w, c])
        return out


class S2Attention(nn.Layer):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.transpose([0, 2, 3, 1])
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = paddle.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.transpose([0, 3, 1, 2])
        return x


if __name__ == '__main__':
    input = paddle.randn([50, 512, 7, 7])
    s2att = S2Attention(channels=512)
    output = s2att(input)
    print(output.shape)
