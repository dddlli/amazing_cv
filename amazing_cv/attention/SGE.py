import numpy as np
import paddle
import paddle.nn as nn
from ..utils.init import *


class SpatialGroupEnhance(nn.Layer):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.weight = paddle.create_parameter([1, groups, 1, 1], dtype='float32')
        self.bias = paddle.create_parameter([1, groups, 1, 1], dtype='float32')
        self.sig = nn.Sigmoid()
        self.init_weights()

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

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape([b * self.groups, -1, h, w])  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(axis=1, keepdim=True)  # bs*g,1,h,w
        t = xn.reshape([b * self.groups, -1])  # bs*g,h*w

        t = t - t.mean(axis=1, keepdim=True)  # bs*g,h*w
        std = t.std(axis=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.reshape([b, self.groups, h, w])  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.reshape([b * self.groups, 1, h, w])  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.reshape([b, c, h, w])

        return x


if __name__ == '__main__':
    input = paddle.randn([50, 512, 7, 7])
    sge = SpatialGroupEnhance(groups=8)
    output = sge(input)
    print(output.shape)
