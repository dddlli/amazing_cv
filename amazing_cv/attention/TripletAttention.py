import paddle
from paddle import nn
from ..utils.basicconv import BasicConv


class ZPool(nn.Layer):
    def forward(self, x):
        
        return paddle.concat((paddle.max(x, 1)[0].unsqueeze(1), paddle.mean(x, 1).unsqueeze(1)), axis=-1)


class AttentionGate(nn.Layer):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = paddle.nn.functional.sigmoid(x_out)
        return x * scale


class TripletAttention(nn.Layer):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.transpose([0, 2, 1, 3])
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.transpose([0, 2, 1, 3])
        x_perm2 = x.transpose([0, 3, 2, 1])
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.transpose([0, 3, 2, 1])
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


if __name__ == '__main__':
    input = paddle.randn((50, 512, 7, 7))
    triplet = TripletAttention()
    output = triplet(input)
    print(output.shape)
