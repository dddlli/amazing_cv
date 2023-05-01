import numpy as np
import paddle
import paddle.nn as nn
from collections import OrderedDict


class SKAttention(nn.Layer):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.LayerList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2D(channel, channel, kernel_size=k, padding=k // 2, groups=group),
                    nn.BatchNorm2D(channel),
                    nn.ReLU())
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.LayerList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(axis=0)

    def forward(self, x):
        bs, c, _, _ = x.shape
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = paddle.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.reshape([bs, c, 1, 1]))  # bs,channel
        attention_weughts = paddle.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


if __name__ == '__main__':
    input = paddle.randn([50, 512, 7, 7])
    se = SKAttention(channel=512, reduction=8)
    print(se)
    output = se(input)
    print(output.shape)
