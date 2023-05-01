import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
from ..utils.init import *


class DoubleAttention(nn.Layer):
    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = nn.Conv2D(in_channels, c_m, 1)
        self.convB = nn.Conv2D(in_channels, c_n, 1)
        self.convV = nn.Conv2D(in_channels, c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2D(c_m, in_channels, kernel_size=1)
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
        assert c == self.in_channels
        A = self.convA(x)  # b,c_m,h,w
        B = self.convB(x)  # b,c_n,h,w
        V = self.convV(x)  # b,c_n,h,w
        tmpA = A.reshape((b, self.c_m, -1))
        attention_maps = F.softmax(B.reshape((b, self.c_n, -1)))
        attention_vectors = F.softmax(V.reshape((b, self.c_n, -1)))
        # step 1: feature gating
        global_descriptors = paddle.bmm(tmpA, attention_maps.transpose((0, 2, 1)))  # b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors)  # b,c_m,h*w
        tmpZ = tmpZ.reshape((b, self.c_m, h, w))  # b,c_m,h,w
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ


if __name__ == '__main__':
    input = paddle.randn([50, 512, 7, 7])
    a2 = DoubleAttention(512, 128, 128, True)
    output = a2(input)
    print(output.shape)
