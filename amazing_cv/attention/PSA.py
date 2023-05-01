import paddle
import paddle.nn as nn
from ..utils.init import *


class PSA(nn.Layer):

    def __init__(self, channel=512, reduction=4, S=4):
        super().__init__()
        self.S = S

        self.convs = []
        for i in range(S):
            self.convs.append(nn.Conv2D(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1))

        self.se_blocks = []
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2D(1),
                nn.Conv2D(channel // S, channel // (S * reduction), kernel_size=1, bias_attr=False),
                nn.ReLU(),
                nn.Conv2D(channel // (S * reduction), channel // S, kernel_size=1, bias_attr=False),
                nn.Sigmoid()
            ))

        self.softmax = nn.Softmax(axis=1)

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

        # Step1:SPC module
        SPC_out = x.reshape([b, self.S, c // self.S, h, w])  # bs,s,ci,h,w
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])

        # Step2:SE weight
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :, :]))
        SE_out = paddle.stack(se_out, axis=1)
        SE_out = SE_out.expand_as(SPC_out)

        # Step3:Softmax
        softmax_out = self.softmax(SE_out)

        # Step4:SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.reshape([b, -1, h, w])

        return PSA_out


if __name__ == '__main__':
    input = paddle.randn([50, 512, 7, 7])
    psa = PSA(channel=512, reduction=8)
    output = psa(input)
    a = output.reshape(-1).sum()
    a.backward()
    print(output.shape)
