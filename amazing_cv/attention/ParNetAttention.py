import paddle
import paddle.nn as nn


class ParNetAttention(nn.Layer):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2D(channel, channel, kernel_size=1),
            nn.BatchNorm2D(channel)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel)
        )
        self.silu = nn.Silu()

    def forward(self, x):
        b, c, _, _ = x.shape
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y


if __name__ == '__main__':
    input = paddle.randn([50, 512, 7, 7])
    pna = ParNetAttention(channel=512)
    output = pna(input)
    print(output.shape)
