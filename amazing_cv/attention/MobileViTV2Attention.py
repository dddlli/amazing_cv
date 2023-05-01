import paddle
import paddle.nn as nn
from ..utils.init import *


class MobileViTV2Attention(nn.Layer):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTV2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model
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
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        i = self.fc_i(x)  # (bs,nq,1)
        weight_i = paddle.nn.functional.softmax(i, axis=1)  # bs,nq,1
        context_score = weight_i * self.fc_k(x)  # bs,nq,d_model
        context_vector = paddle.sum(context_score, axis=1, keepdim=True)  # bs,1,d_model
        v = self.fc_v(x) * context_vector  # bs,nq,d_model
        out = self.fc_o(v)  # bs,nq,d_model

        return out


if __name__ == '__main__':
    input = paddle.randn([50, 49, 512])
    sa = MobileViTV2Attention(d_model=512)
    output = sa(input)
    print(output.shape)
