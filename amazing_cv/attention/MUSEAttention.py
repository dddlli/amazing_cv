import numpy as np
import paddle
import paddle.nn as nn
from ..utils.init import *


class Depth_Pointwise_Conv1d(nn.Layer):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        if (k == 1):
            self.depth_conv = nn.Identity()
        else:
            self.depth_conv = nn.Conv1D(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k // 2
            )
        self.pointwise_conv = nn.Conv1D(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1
        )

    def forward(self, x):
        out = self.pointwise_conv(self.depth_conv(x))
        return out


class MUSEAttention(nn.Layer):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = Depth_Pointwise_Conv1d(h * d_v, d_model, 1)
        self.conv3 = Depth_Pointwise_Conv1d(h * d_v, d_model, 3)
        self.conv5 = Depth_Pointwise_Conv1d(h * d_v, d_model, 5)
        self.dy_paras = paddle.create_parameter(shape=[3], dtype='float32')
        self.softmax = nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.named_sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                zeros_(m.weight)
                ones_(m.bias)
            elif isinstance(m, nn.Linear):
                normal_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        # Self Attention
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).reshape([b_s, nq, self.h, self.d_k]).transpose([0, 2, 1, 3])  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).reshape([b_s, nk, self.h, self.d_k]).transpose([0, 2, 3, 1])  # (b_s, h, d_k, nk)
        v = self.fc_v(values).reshape([b_s, nk, self.h, self.d_v]).transpose([0, 2, 1, 3])  # (b_s, h, nk, d_v)

        att = paddle.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = paddle.nn.functional.softmax(att, -1)
        att = self.dropout(att)

        out = paddle.matmul(att, v).transpose([0, 2, 1, 3]).reshape([b_s, nq, self.h * self.d_v])  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        v2 = v.transpose([0, 1, 3, 2]).reshape([b_s, -1, nk])  # bs,dim,n
        self.dy_paras = paddle.create_parameter(shape=[3], dtype='float32')
        out2 = self.dy_paras[0] * self.conv1(v2) + self.dy_paras[1] * self.conv3(v2) + self.dy_paras[2] * self.conv5(v2)
        out2 = out2.transpose([0, 2, 1])  # bs.n.dim

        out = out + out2
        return out


if __name__ == '__main__':
    input = paddle.randn([50, 49, 512])
    sa = MUSEAttention(d_model=512, d_k=512, d_v=512, h=8)
    output = sa(input, input, input)
    print(output.shape)
