import numpy as np
import paddle
from paddle import nn
from ..utils.init import *


class SEAttention(nn.Layer):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias_attr=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias_attr=False),
            nn.Sigmoid()
        )

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
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape([b, c])
        y = self.fc(y).reshape([b, c, 1, 1])
        return x * y.expand_as(x)


if __name__ == '__main__':
    input = paddle.randn([50, 512, 7, 7])
    se = SEAttention(channel=512, reduction=8)
    output = se(input)
    print(output.shape)
