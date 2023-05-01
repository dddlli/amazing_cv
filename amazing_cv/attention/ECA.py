import numpy as np
import paddle
from paddle import nn
from ..utils.init import *


class ECAAttention(nn.Layer):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
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

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).transpose([0, 2, 1])  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.transpose([0, 2, 1]).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)


if __name__ == '__main__':
    input = paddle.randn((50, 512, 7, 7))
    eca = ECAAttention(kernel_size=3)
    output = eca(input)
    print(output.shape)
