import numpy as np
import paddle
from paddle import nn


class ResidualAttention(nn.Layer):

    def __init__(self, channel=512, num_class=1000, la=0.2):
        super().__init__()
        self.la = la
        self.fc = nn.Conv2D(in_channels=channel, out_channels=num_class, kernel_size=1, stride=1, bias_attr=False)

    def forward(self, x):
        b, c, h, w = x.shape
        y_raw = self.fc(x).flatten(2)  # b,num_class,hxw
        y_avg = paddle.mean(y_raw, axis=2)  # b,num_class
        y_max = paddle.max(y_raw, axis=2)[0]  # b,num_class
        score = y_avg + self.la * y_max
        return score


if __name__ == '__main__':
    input = paddle.randn([50, 512, 7, 7])
    resatt = ResidualAttention(channel=512, num_class=1000, la=0.2)
    output = resatt(input)
    print(output.shape)
